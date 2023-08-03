from logging import Logger
from connection import HTTPLIB2Capture, ModelPredictionsReceiver
from min_max_utils.min_max_utils import filter_boxes, check_box_in_area, convert_coords_from_dict_to_list, drop_area, \
    most_common
from confs.load_configs import N_STEPS
from min_max_utils.MinMaxReporter import Reporter
import time
from typing import Sequence


def run_min_max(dataset: HTTPLIB2Capture, logger: Logger, areas: Sequence[dict],
                folder: str, debug_folder: str, server_url: str, zones: list):
    
    model_pred_receiver = ModelPredictionsReceiver(server_url, logger)
    stat_history = []
    is_human_was_detected = True
    reporter = Reporter(logger, server_url, folder, debug_folder)

    while True:
        img = dataset.get_snapshot()
        if img is None:
            time.sleep(1)
            continue

        logger.debug("Sending request to model server")
        human_preds = model_pred_receiver.predict_human(img.copy())
        logger.debug("Human preds received")

        is_human_in_image_now = human_preds is not None and human_preds.size 

        if is_human_in_image_now:
            logger.debug("Human is detected")
            time.sleep(1)
            is_human_was_detected = is_human_in_image_now
            continue

        if (is_human_was_detected and not is_human_in_image_now) or \
                (not is_human_was_detected and not is_human_in_image_now and
                 len(stat_history)):
            logger.debug("Objects counting...")
            if zones:
                cropped_images = []
                for zone in zones:
                    x1, y1, x2, y2 = convert_coords_from_dict_to_list(zone.get("coords")[0])
                    cropped_images.append(img[y1:y2, x1:x2])    
                model_preds_boxes = [model_pred_receiver.predict_boxes(crop_img) for crop_img in cropped_images]
                if any([elem is None for elem in model_preds_boxes]):
                    continue
                model_preds_bottles = [model_pred_receiver.predict_bottles(crop_img) for crop_img in cropped_images]
                if any([elem is None for elem in model_preds_bottles]):
                    continue

        areas_stat = []

        for area_index, item in enumerate(areas.copy()):

            item_stat = []
            for subarr_idx, coord in enumerate(item['coords'].copy()):
                area_coord = convert_coords_from_dict_to_list(coord)

                if area_coord[0] == area_coord[2] or area_coord[1] == area_coord[3]:
                    logger.warning("Empty area")
                    drop_area(areas, area_index, item, subarr_idx)
                    continue

                if is_human_in_image_now:
                    stat_history.clear()
                    areas_stat.clear()

                if (is_human_was_detected and not is_human_in_image_now) or \
                        (not is_human_was_detected and not is_human_in_image_now and len(stat_history)):
                    boxes_preds = None
                    if zones:
                        model_preds_to_use = model_preds_boxes if item["task"] in ["boxes", "box"] else model_preds_bottles
                        for idx, zone in enumerate(zones):
                            zone_coords = convert_coords_from_dict_to_list(zone.get("coords")[0])
                            if check_box_in_area(area_coord, zone_coords):
                                boxes_preds = filter_boxes(
                                    zone_coords, 
                                    model_preds_to_use[idx], 
                                    area_coord
                                )
                                logger.debug(f"{item.get('itemName')} item -> {zone.get('zoneName')} zone")
                                item["zoneId"] = zone.get("zoneId")
                                break
                        if boxes_preds is None:
                            logger.critical("Area is not in zone")
                            exit(1)
                    else:
                        send_request_func = model_pred_receiver.predict_boxes if item["task"] == "boxes" else model_pred_receiver.predict_bottles
                        model_preds = send_request_func(img[area_coord[1]:area_coord[3], area_coord[0]:area_coord[2]])
                        if model_preds is None:
                            time.sleep(1)
                            break
                        boxes_preds = filter_boxes(area_coord, model_preds, check=False)
                    item_stat.append(boxes_preds)
            if item_stat:
                areas_stat.append(item_stat)

        is_human_was_detected = is_human_in_image_now

        if len(areas_stat) >= len(areas) and not is_human_in_image_now:
            stat_history.append(areas_stat.copy())

        areas_stat.clear()

        if len(stat_history) >= N_STEPS and not is_human_in_image_now:
            logger.debug("Objects number matrix creation")
            n_boxes_per_area = []
            coords_per_area = []
            for item_idx, item_iter in enumerate(stat_history[0]):
                n_box_item_ctxt = []
                coord_item_ctxt = []
                for arr_idx, arr in enumerate(item_iter):
                    arr_n_hist = []
                    for tmp_st_idx in range(len(stat_history)):
                        arr_n_hist.append(stat_history[tmp_st_idx][item_idx][arr_idx][0])
                    msc_n = most_common(arr_n_hist)
                    idx = 0
                    while len(stat_history[idx][item_idx][arr_idx][1]) != msc_n:
                        idx += 1
                    n_box_item_ctxt.append(msc_n)
                    coord_item_ctxt.append(stat_history[idx][item_idx][arr_idx][1])
                n_boxes_per_area.append(n_box_item_ctxt)
                coords_per_area.append(coord_item_ctxt)
            logger.debug("Report creation")
            reporter.send_report_to_server(
                reporter.create_report(n_boxes_per_area, img, areas, coords_per_area, zones))
            stat_history.clear()
            logger.debug("Report sent")
