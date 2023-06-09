from min_max_utils.HTTPLIB2Capture import HTTPLIB2Capture
from logging import Logger
from ultralytics import YOLO
from min_max_utils.min_max_utils import *
from confs.load_configs import N_STEPS

def run_min_max(dataset: HTTPLIB2Capture, logger: Logger, human_model: YOLO, box_model: YOLO, areas:list[dict], folder: str, server_url: str, stelag_coords: list[list]):
    stat_history = []
    is_human_was_detected = True
    n_iters = 0

    while True:
        img = dataset.get_snapshot()
        if img is None:
            logger.warning("Empty image")
            continue
        cropped_img = img[main_item_coords[1]:main_item_coords[3], main_item_coords[0]:main_item_coords[2]]
        n_iters += 1
        if n_iters % 60 == 0:
            logger.debug("60 detect iterations passed")

        is_human_in_area_now = human_model(cropped_img.copy())[0] != 0

        if (is_human_was_detected and not is_human_in_area_now) or \
            (not is_human_was_detected and not is_human_in_area_now and \
                        len(stat_history)):
            logger.debug("Boxes counting...")
            model_preds = box_model(cropped_img)

        if is_human_in_area_now:
            logger.debug("Human was detected")

        areas_stat = []
        n_items = 0

        for area_index, item in enumerate(areas.copy()):  # for each area
            counter = 0

            n_items += len(item['coords'])
            item_stat = []
            for subarr_idx, coord in enumerate(item['coords'].copy()):
                x1, y1, x2, y2 = list(
                    map(round, (coord['x1'], coord['y1'], coord['x2'], coord['y2'])))
                if x1 == x2 or y1 == y2:
                    logger.warning("Empty area")
                    n_items -= 1
                    drop_area(areas, area_index, item, subarr_idx)
                    continue
                
                if is_human_was_detected and is_human_in_area_now:  # wait for human disappear
                    stat_history.clear()
                    areas_stat.clear()

                elif not is_human_was_detected and is_human_in_area_now:  
                    stat_history.clear()
                    areas_stat.clear()

                if (is_human_was_detected and not is_human_in_area_now) or\
                    (not is_human_was_detected and not is_human_in_area_now and len(stat_history)) : 
                    filtered_boxes = filter_boxes([x1, y1, x2, y2], main_item_coords, *model_preds)
                    item_stat.append(filtered_boxes)

            if item_stat:
                areas_stat.append(item_stat)

        is_human_was_detected = is_human_in_area_now

        if len(areas_stat) >= len(areas):
            stat_history.append(areas_stat.copy())

        areas_stat.clear()

        if len(stat_history) >= N_STEPS:   # n_steps x n_items x n_subarrs x 2
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
                        folder, logger, server_url, coords_per_area, main_item_coords)
            stat_history.clear()
