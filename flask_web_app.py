from flask import Flask, render_template, send_file, request
import yaml
from flask import jsonify

app = Flask(__name__)

WEB_SEAT_COLOR = {
    "red__": (255, 0, 0),
    "green": (0, 255, 0),
    "blue_": (0, 0, 255)
}

def WEB_get_empty_D_template_for_seat():
    return {
        "tag_name": "scene_NONE",
        "number_of_tables": -1,
        "table_layout": (-1, -1),
        "seats_color": [],
    }

def write_tags_to_yaml(tags):
    """
    expected tags format:
    # example  tag
    tags__ = [ {'tag_name': "scene_1",
              "number_of_tables": 2,
              "table_layout": (2, 4),
              "seats_color": [[green, blue_, green, blue_, blue_, green, blue_, blue_],
                              [red__, red__, red__, red__, red__, red__, red__, red__]],
              },
             ## example 2
             {'tag_name': "scene_2",
              "number_of_tables": 3,
              "table_layout": (2, 3),
              "seats_color": [[green, blue_, green, blue_, blue_, green],
                              [red__, red__, red__, red__, red__, red__],
                              [red__, red__, red__, red__, red__, red__],
                              ],
              },
             ## example 3
             {'tag_name': "scene_3",
              "number_of_tables": 4,
              "table_layout": (2, 2),
              "seats_color": [[green, blue_, green, blue_],
                              [red__, red__, red__, red__],
                              [red__, red__, red__, red__],
                              [green, green, green, green],
                              ],
              },
           ]

    :param tags:
    :return:
    """

    with open('./web_datas/tags.yaml', 'w') as file:
        yaml.dump(tags, file)

def read_tags_from_yaml():
    with open('./web_datas/tags.yaml', 'r') as file:
        tags = yaml.load(file, Loader=yaml.FullLoader)
    return tags

@app.route('/seat_status')
def seat_status():
    scene = request.args.get('scene', 'default')
    tags = read_tags_from_yaml()
    # 找到對應的場景數據
    tag = next((item for item in tags if item["tag_name"] == scene), None)
    if tag:
        return jsonify(tag)
    else:
        return jsonify({"error": "Scene not found"}), 404

@app.route('/')
def index():
    blue_ = WEB_SEAT_COLOR["blue_"]
    green = WEB_SEAT_COLOR["green"]
    red__ = WEB_SEAT_COLOR["red__"]
    tags = read_tags_from_yaml()

    # 計算每個場景的座位狀態
    for tag in tags:
        total_seats = sum(len(table) for table in tag['seats_color'])
        available_seats = sum(seat == green for table in tag['seats_color'] for seat in table)
        occupied_seats = sum(seat == red__ for table in tag['seats_color'] for seat in table)
        avaiable_seats = sum(seat == blue_ for table in tag['seats_color'] for seat in table)
        tag['total_seats'] = total_seats
        tag['available_seats'] = available_seats
        tag['occupied_seats'] = occupied_seats
        tag['avaiable_seats'] = avaiable_seats

    return render_template('index.html', tags=tags)


@app.route('/live_feed')
def live_feed():
    scene = request.args.get('scene', 'default')
    image_path = f"./web_datas/scene_real_time/{scene}.jpg"
    print(f"image_path = '{image_path}'")
    return send_file(image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)