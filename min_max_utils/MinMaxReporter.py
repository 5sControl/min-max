import requests
import numpy as np
from logging import Logger
from min_max_utils.min_max_utils import find_red_line, convert_coords_from_dict_to_list, is_line_in_area
from min_max_utils.visualization_utils import draw_line, draw_rect, draw_text
from min_max_utils.img_process_utils import save_image
import uuid
import datetime
import os


class Reporter:
    def __init__(self, logger: Logger, server_url: str, user_folder: str, debug_folder: str) -> None:
        self.logger = logger
        self.server_url = server_url
        self.user_folder = user_folder
        if not os.path.exists(self.user_folder):
            os.makedirs(self.user_folder)
        self.debug_folder = debug_folder
        if not os.path.exists(self.debug_folder):
            os.makedirs(self.debug_folder)

    def add_empty_zone(self, zones: list):
        empty_zone =                 {
                    "zoneId": None,
                    "zoneName": None,
                    "coords":
                        [{
                            "x1": 0,
                            "x2": 1920,
                            "y1": 0,
                            "y2": 1080
                    }],
                    "items": []
                }
        
        zones.append(empty_zone)
        return zones

    def create_report(self, n_boxes_history: list, img: np.array, areas: list, boxes_coords: list, zones: list) -> str:
        red_lines = find_red_line(img)
        report = []
        if not zones:
            for item in areas:
                item['zoneId'] = None
            zones = self.add_empty_zone(zones)
        for zone in zones:
            zone_dict = {'zoneId': zone['zoneId'], 'zoneName': zone['zoneName'], 'items': []}
            report.append(zone_dict)
        for item_index, item in enumerate(areas):
            itemid = item['itemId']

            item_name = item['itemName']
            user_dbg_image_name_url = self.user_folder + '/' + str(uuid.uuid4()) + '.png'
            
            debug_user_image = img.copy()

            rectangle_color = (0, 102, 204)
            for zone in zones:
                debug_user_image = draw_rect(debug_user_image, convert_coords_from_dict_to_list(zone.get("coords")[0]), rectangle_color)

            is_red_line_in_item = False

            for subarr_idx, coord in enumerate(item['coords']):
                area_coords = convert_coords_from_dict_to_list(coord)

                is_red_line_in_subarea = False

                for line in red_lines:
                    if is_line_in_area(area_coords, line):
                        debug_user_image = draw_line(debug_user_image, line, area_coords, thickness=4)
                        is_red_line_in_subarea = is_red_line_in_item = True

                text_item = f"{item_name}: {n_boxes_history[item_index][subarr_idx] if not is_red_line_in_subarea else 'low stock level'}"

                debug_user_image = draw_rect(debug_user_image, area_coords, rectangle_color, thickness=2)

                for idx, bbox_coords in enumerate(boxes_coords[item_index][subarr_idx]):
                    text = str(idx + 1)

                    debug_user_image = draw_rect(
                        debug_user_image, 
                        bbox_coords[:4], 
                        (51, 51, 255) if is_red_line_in_item else (0, 255, 0), 
                        thickness=2
                    )
                    debug_user_image = draw_text(
                        debug_user_image,
                        bbox_coords[:2],
                        text, 
                        bbox_coords[2] - bbox_coords[0], 
                        (255, 255, 255),
                        min_font_size=15, 
                        img_fraction=None
                    )


                debug_user_image = draw_text(
                    debug_user_image, 
                    area_coords[:2], 
                    text_item, 
                    area_coords[2] - area_coords[0], 
                    (255, 255, 255)
                )

            status_text = 'Out of stock' if is_red_line_in_item else 'In stock'
            text_all_img = f"{status_text}\n{text_item} total number: {sum(n_boxes_history[item_index])}"
            debug_user_image = draw_text(
                    debug_user_image, 
                    (0, 0), 
                    text_all_img, 
                    1920, 
                    (255, 255, 255),
                    img_fraction=0.4
            )

            for idx, zone in enumerate(zones):
                if zone.get("zoneId") == item.get("zoneId"):
                    report[idx]['items'].append(
                        {
                            "itemId": itemid,
                            "count": sum(n_boxes_history[item_index]),
                            "image_item": user_dbg_image_name_url,
                            "low_stock_level": is_red_line_in_item,
                            "zoneId": item.get("zoneId"),
                            "zoneName": zone.get("zoneName")
                        }
                    )
            save_image(debug_user_image, user_dbg_image_name_url)
        photo_start = {
            'date': datetime.datetime.now()
        }
        report_for_send = {
            'camera': os.path.basename(self.user_folder),
            'algorithm': "min_max_control",
            'start_tracking': str(photo_start['date']),
            'stop_tracking': str(photo_start['date']),
            'photos': [{'date': str(photo_start['date'])}],
            'violation_found': False,
            'extra': report
        }
        return report_for_send

    def send_report_to_server(self, report: dict) -> None:
        self.logger.info(
            '\n'.join(['<<<<<<<<<<<<<<<<<SEND REPORT!!!!!!!>>>>>>>>>>>>>>',
                    str(report),
                    f'{self.server_url}:8000/api/reports/report-with-photos/'])
        )
        try:
            requests.post(url=f'{self.server_url}:80/api/reports/report-with-photos/', json=report)
        except Exception as exc:
            self.logger.error("Error while sending report occurred: {}".format(exc))