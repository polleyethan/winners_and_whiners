import tensorflow as tf
import numpy as np
import json
from espn_page_parsing import getPassableStatArray

points_model = tf.keras.models.load_model('models/points_models/model1')
win_model = tf.keras.models.load_model('models/win_models/model1')
def print_results(home_team, away_team, points_model, win_model, exclude_dtd):

    game_to_pass = getPassableStatArray(home_team["id"], away_team["id"], home_team["abbrev"], away_team["abbrev"], exclude_dtd)

    game = tf.convert_to_tensor(game_to_pass)
    print(game.shape)
    game = tf.reshape(game, (1,636))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/test")

    point_call = points_model.predict(game)
    win_call = win_model.predict(game, callbacks=[tensorboard])

    print(point_call)
    print(win_call)

    print("****** Score PREDICTION ******")
    print("Home ("+home_team["abbrev_show"]+"): "+str(point_call[0][0]))
    print("Away ("+away_team["abbrev_show"]+"): "+str(point_call[0][1]))

    print("****** WIN PREDICTION ******")
    print("Home ("+home_team["abbrev_show"]+"): "+str(win_call[0][0]*100)+"%")
    print("Away ("+away_team["abbrev_show"]+"): "+str(win_call[0][1]*100)+"%")


points_model = tf.keras.models.load_model('models/points_models/model1')
win_model = tf.keras.models.load_model('models/win_models/model1')

knicks = {"id": "1610612752", "abbrev": "ny", "abbrev_show":"NYK"}
nets = {"id": "1610612751", "abbrev": "bkn", "abbrev_show":"BKN"}
pacers = {"id": "1610612754", "abbrev": "ind", "abbrev_show":"IND"}
sixers = {"id": "1610612755", "abbrev": "phi", "abbrev_show":"PHI"}
hornets = {"id": "1610612766", "abbrev": "cha", "abbrev_show":"CHA"}
bulls = {"id": "1610612741", "abbrev": "chi", "abbrev_show":"CHI"}
cavs = {"id": "1610612739", "abbrev": "cle", "abbrev_show":"CLE"}
wizards = {"id": "1610612764", "abbrev": "wsh", "abbrev_show":"WSH"}
pistons = {"id": "1610612765", "abbrev": "det", "abbrev_show":"DET"}
bucks = {"id": "1610612749", "abbrev": "mil", "abbrev_show":"MIL"}
heat = {"id": "1610612748", "abbrev": "mia", "abbrev_show":"MIA"}
thunder = {"id": "1610612760", "abbrev": "okc", "abbrev_show":"OKC"}
raptors = {"id": "1610612761", "abbrev": "tor", "abbrev_show":"TOR"}
rockets = {"id": "1610612745", "abbrev": "hou", "abbrev_show":"HOU"}
mavs = {"id": "1610612742", "abbrev": "dal", "abbrev_show":"DAL"}
grizzlies = {"id": "1610612763", "abbrev": "mem", "abbrev_show":"MEM"}
jazz = {"id": "1610612762", "abbrev": "utah", "abbrev_show":"UTAH"}
twolves = {"id": "1610612750", "abbrev": "min", "abbrev_show":"MIN"}
nuggets = {"id": "1610612743", "abbrev": "den", "abbrev_show":"DEN"}
pelicans = {"id": "1610612740", "abbrev": "no", "abbrev_show":"NOP"}
blazers = {"id": "1610612757", "abbrev": "por", "abbrev_show":"POR"}
warriors = {"id": "1610612744", "abbrev": "gs", "abbrev_show":"GSW"}
magic = {"id": "1610612753", "abbrev": "orl", "abbrev_show":"ORL"}
kings = {"id": "1610612758", "abbrev": "sac", "abbrev_show":"SAC"}
celtics = {"id": "1610612738", "abbrev": "bos", "abbrev_show":"BOS"}
clippers = {"id": "1610612746", "abbrev": "lac", "abbrev_show":"LAC"}
lakers = {"id": "1610612747", "abbrev": "lal", "abbrev_show":"LAL"}
spurs = {"id": "1610612759", "abbrev": "sa", "abbrev_show":"SA"}

matchups = [[spurs, nuggets]]

for g in matchups:
    print_results(g[0], g[1], points_model, win_model, True)
""""

knicks: 1610612752, ny
spurs: 1610612759, sa
nets: 1610612751, bkn
mavericks: 1610612742, dal
celtics: 1610612738, bos
lakers: 1610612747, lal

team_data = {"id": "id", "abbrev": "abbrev", "abbrev_show":"abbrev_show"}
"""