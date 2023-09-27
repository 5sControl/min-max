from logging import Logger
from connection import ImageCapture, ModelPredictionsReceiver
from min_max_utils.min_max_utils import filter_boxes, check_box_in_area, convert_coords_from_dict_to_list, drop_area, \
    most_common
from confs.load_configs import configs
from min_max_utils.MinMaxReporter import Reporter
import time
import numpy as np
from typing import Sequence


class MinMaxAlgorithm:
    def __init__(self, http_capture: ImageCapture, logger: Logger, areas: Sequence[dict],
                folder: str, server_url: str, zones: list) -> None:
        self._http_capture = http_capture
        self._logger = logger
        self._areas = areas
        self._folder = folder
        self._server_url = server_url
        self._zones = zones
        self._model_preds_receiver =  ModelPredictionsReceiver(self._server_url, self._logger)
        self._step_count_history = []
        self._is_human_was_detected = True
        self._reporter = Reporter(self._logger, self._server_url, self._folder)
        self._min_epoch_time = configs["min_epoch_time"]
        self._ssim_threshold = configs["ssim_threshold"]
        self._n_steps = configs["n_steps"]
        self._first_report = True

    def _check_if_call_models(self, is_human_in_image_now: bool) -> bool:
        # check if situation is appropriate for calling models
        return (self._is_human_was_detected and not is_human_in_image_now) or \
               (not self._is_human_was_detected and not is_human_in_image_now and len(self._step_count_history))
    
    def _add_zone_id_key_for_items(self):
        for item_idx, item in enumerate(self._areas):
            item_coords = convert_coords_from_dict_to_list(item["coords"][0])
            for zone in self._zones:
                if check_box_in_area(item_coords, convert_coords_from_dict_to_list(zone["coords"][0])):
                    self._areas[item_idx]["zoneId"] = zone["zoneId"]
                    
    def _crop_image_by_zones(self, image: np.array, zones: dict) -> dict:
        cropped_images = {}
        for zone in zones:
            x1, y1, x2, y2 = convert_coords_from_dict_to_list(zone["coords"][0])
            print( convert_coords_from_dict_to_list(zone["coords"][0]))
            cropped_images[zone['zoneId']] = image[y1:y2, x1:x2]
        return cropped_images
    
    def _add_count_to_history(self):
        pass

    def _clear_count_history(self):
        self._step_count_history.clear()

    def _add_main_zone(self):
        min_y = min_x = 2000
        max_y = max_x = -1
        for item in self._areas:
            for coordinates in item["coords"]:
                min_x = coordinates['x1'] if coordinates['x1'] < min_x else min_x
                min_y = coordinates['y1'] if coordinates['y1'] < min_y else min_y
                max_x = coordinates['x2'] if coordinates['x2'] > max_x else max_x
                max_y = coordinates['y2'] if coordinates['y2'] > max_y else max_y
        self._zones = [{'zoneId': 1, 'zoneName': 'main', 'coords': [{'x1': min_x, 'x2': max_x, 'y1': min_y, 'y2': max_y}]}]

    def start(self) -> None:
        self._add_main_zone()
        self._add_zone_id_key_for_items()
        while True:
            start_epoch_time = time.time()
            self._run_one_min_max_epoch()
            end_epoch_time = time.time()
            passed_time = end_epoch_time - start_epoch_time
            if passed_time < self._min_epoch_time:
                time.sleep(self._min_epoch_time - passed_time)

    def _run_one_min_max_epoch(self) -> None:
        image, ssim_value = self._http_capture.get_snapshot()
        if image is None:
            return
        if not self._first_report and not self._is_human_was_detected and ssim_value > self._ssim_threshold:
            self._logger.info("Similar images. Skipping iteration...")
            return
        self._logger.debug("Sending request to model server")
        human_preds = self._model_preds_receiver.predict_human(image.copy())
        if human_preds is None:
            return
        self._logger.debug("Human preds received")

        is_human_in_image_now = human_preds is not None and human_preds.size

        if is_human_in_image_now:
            # skip this iteration
            self._clear_count_history()
            self._logger.debug("Human is detected")
            self._is_human_was_detected = True
            return

        if self._check_if_call_models(is_human_in_image_now):
            if self._zones:
                self._logger.debug("Object counting in zones mode...")
                zone_img_fragments = self._crop_image_by_zones(image, self._zones)
                model_preds_boxes = {
                    zone_key:self._model_preds_receiver.predict_boxes(img_fragment) \
                        for (zone_key, img_fragment) in zone_img_fragments.items()
                }
                if any([val is None for _, val in model_preds_boxes.items()]):
                    self._is_human_was_detected = is_human_in_image_now
                    return
                model_preds_bottles = {
                    zone_key:self._model_preds_receiver.predict_bottles(img_fragment) \
                        for (zone_key, img_fragment) in zone_img_fragments.items()
                }
                if any([val is None for _, val in model_preds_bottles.items()]):
                    self._is_human_was_detected = is_human_in_image_now
                    return
            else:
                self._logger.fatal("At least one zone must be in zones variables")
                exit(1)
        
        step_count_stat = []
        
        for item_idx, item in enumerate(self._areas.copy()):
            item_count_stat = []
            for subarr_idx, subarr_coord in enumerate(item['coords'].copy()):
                subarr_coord = convert_coords_from_dict_to_list(subarr_coord)
                if subarr_coord[0] == subarr_coord[2] or subarr_coord[1] == subarr_coord[3]:
                    self._logger.warning("Empty area")
                    drop_area(self._areas, item_idx, item, subarr_idx)
                    return
                if self._check_if_call_models(is_human_in_image_now):
                    if self._zones:
                        model_preds_to_use = model_preds_bottles if "bottle" in item["task"] else model_preds_boxes
                        zone_id = item["zoneId"]
                        for idx, zone in enumerate(self._zones):
                            if zone["zoneId"] == zone_id:
                                zone_coords = convert_coords_from_dict_to_list(zone["coords"][0])
                                break
                            if idx == len(self._zones) - 1:
                                self._logger.fatal(f"Unknown zone id - {zone_id}")
                        boxes_preds = filter_boxes(
                            zone_coords,
                            model_preds_to_use[zone_id],
                            subarr_coord
                        )
                    item_count_stat.append(boxes_preds)
            if item_count_stat:
                step_count_stat.append(item_count_stat)
        self._is_human_was_detected = is_human_in_image_now
        if len(step_count_stat) == len(self._areas):
            self._step_count_history.append(step_count_stat.copy())

        if len(self._step_count_history) == self._n_steps:
            self._logger.debug("Objects number matrix creation")
            self._send_data_for_report(image)

    def _send_data_for_report(self, image):
        n_boxes_per_area = []
        coords_per_area = []
        self._logger.debug(f"N BOXES HIST PER STEP >>>\n {self._step_count_history} \n<<<")
        for item_idx, item_iter in enumerate(self._step_count_history[0]):
                n_box_item_ctxt = []
                coord_item_ctxt = []
                for arr_idx, arr in enumerate(item_iter):
                    arr_n_hist = []
                    for tmp_st_idx in range(len(self._step_count_history)):
                        arr_n_hist.append(self._step_count_history[tmp_st_idx][item_idx][arr_idx][0])
                    msc_n = most_common(arr_n_hist)
                    idx = 0
                    while len(self._step_count_history[idx][item_idx][arr_idx][1]) != msc_n:
                        idx += 1
                    n_box_item_ctxt.append(msc_n)
                    coord_item_ctxt.append(self._step_count_history[idx][item_idx][arr_idx][1])
                n_boxes_per_area.append(n_box_item_ctxt)
                coords_per_area.append(coord_item_ctxt)
        self._logger.debug("Report creation")
        self._reporter.send_report_to_server(
        self._reporter.create_report(n_boxes_per_area, image, self._areas, coords_per_area, self._zones.copy()))
        self._step_count_history.clear()
        self._logger.debug("Report sent")
        self._first_report = False
