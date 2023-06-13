from min_max_utils.HTTPLIB2Capture import HTTPLIB2Capture
from logging import Logger
from ultralytics import YOLO
from min_max_utils.min_max_utils import filter_boxes, check_box_in_area, convert_coords_from_dict_to_list, drop_area, \
    most_common, send_report
from confs.load_configs import N_STEPS


def run_min_max(dataset: HTTPLIB2Capture, logger: Logger, human_model: YOLO, box_model: YOLO, areas: list[dict],
                folder: str, server_url: str, zones: list):
    stat_history = []
    is_human_was_detected = True
    n_iters = 0

    while True:
        img = dataset.get_snapshot()
        if img is None:
            logger.warning("Empty image")
            continue
        n_iters += 1
        if n_iters % 60 == 0:
            logger.debug("60 detect iterations passed")

        is_human_in_area_now = human_model(img.copy())[0] != 0

        if (is_human_was_detected and not is_human_in_area_now) or \
                (not is_human_was_detected and not is_human_in_area_now and
                 len(stat_history)):
            logger.debug("Boxes counting...")
            if zones:
                cropped_images = []
                for zone in zones:
                    x1, x2, y1, y2 = tuple(zone.get("coords").values())
                    cropped_images.append(img[y1:y2, x1:x2])
                model_preds = [box_model(crop_img) for crop_img in cropped_images]

        if is_human_in_area_now:
            logger.debug("Human was detected")

        areas_stat = []

        for area_index, item in enumerate(areas.copy()):

            item_stat = []
            for subarr_idx, coord in enumerate(item['coords'].copy()):
                area_coord = convert_coords_from_dict_to_list(coord)

                if area_coord[0] == area_coord[2] or area_coord[1] == area_coord[3]:
                    logger.warning("Empty area")
                    drop_area(areas, area_index, item, subarr_idx)
                    continue

                if (is_human_was_detected and is_human_in_area_now) or \
                        (not is_human_was_detected and is_human_in_area_now):
                    stat_history.clear()
                    areas_stat.clear()

                if (is_human_was_detected and not is_human_in_area_now) or \
                        (not is_human_was_detected and not is_human_in_area_now and len(stat_history)):
                    boxes_preds = None
                    if zones:
                        for idx, zone in enumerate(zones):
                            zone_coords = convert_coords_from_dict_to_list(zone.get("coords"))
                            if check_box_in_area(area_coord, zone_coords):
                                boxes_preds = filter_boxes(zone_coords, *model_preds[idx], area_coord)
                                item["zoneId"] = zone.get("zoneId")
                                break
                        if boxes_preds is None:
                            logger.critical("Area is not in zone")
                            exit(1)
                    else:
                        model_preds = box_model(img[area_coord[1]:area_coord[3], area_coord[0]:area_coord[2]])
                        boxes_preds = filter_boxes(area_coord, *model_preds, check=False)
                    item_stat.append(boxes_preds)
            if item_stat:
                areas_stat.append(item_stat)

        is_human_was_detected = is_human_in_area_now

        if len(areas_stat) >= len(areas):
            stat_history.append(areas_stat.copy())

        areas_stat.clear()

        if len(stat_history) >= N_STEPS:
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
            send_report(n_boxes_per_area, img, areas,
                        folder, logger, server_url, coords_per_area, zones)
            stat_history.clear()
