import http.client
import json
from datetime import datetime
from datetime import date
from dateutil import parser
import pytz
import time
import threading
import queue

conn = http.client.HTTPSConnection("api.sportradar.us")
conn2 = http.client.HTTPSConnection("api.sportradar.us")
conn3 = http.client.HTTPSConnection("api.sportradar.us")
conn4 = http.client.HTTPSConnection("api.sportradar.us")
conn5 = http.client.HTTPSConnection("api.sportradar.us")

# EXAMPLE RUN:
#       data = getScheduleForYear(key, 2019)
# DATA EXAMPLE:
#       sample_data/schedule_sample.json
def getScheduleForYear(apikey, year):

    conn.request("GET", "/nba/trial/v7/en/games/"+str(year)+"/REG/schedule.json?api_key="+apikey)

    res = conn.getresponse()
    data = res.read()

    data_decoded = data.decode("utf-8")

    formatted = json.loads(data_decoded)
    return formatted['games']


# EXAMPLE RUN:
#       data = getGameDataById(key, "c5f0698c-db24-428f-8160-52fe2b67fa08")
# DATA EXAMPLE:
#       sample_data/game_stats_sample.json
def getGameDataById(apikey, gameId, connnum):
    if connnum == 1:
        conn.request("GET", "/nba/trial/v5/en/games/"+gameId+"/summary.json?api_key="+apikey)
        res = conn.getresponse()
    elif connnum == 2:
        conn2.request("GET", "/nba/trial/v7/en/games/"+gameId+"/summary.json?api_key="+apikey)
        res = conn2.getresponse()
    elif connnum == 3:
        conn3.request("GET", "/nba/trial/v7/en/games/"+gameId+"/summary.json?api_key="+apikey)
        res = conn3.getresponse()
    elif connnum == 4:
        conn4.request("GET", "/nba/trial/v7/en/games/"+gameId+"/summary.json?api_key="+apikey)
        res = conn4.getresponse()
    elif connnum == 5:
        conn5.request("GET", "/nba/trial/v7/en/games/"+gameId+"/summary.json?api_key="+apikey)
        res = conn5.getresponse()
    data = res.read()

    decoded = data.decode("utf-8")
    formatted = json.loads(decoded)
    return formatted


# EXAMPLE RUN:
#       data = getTeamScheduleByAlias(key, "NYK", 2021)
# DATA EXAMPLE:
#       sample_data/team_schedule_sample.json.json
def getTeamScheduleByAlias(game_data, alias):

    team_filter = [x for x in game_data if x["home"]["alias"] == alias or x["away"]["alias"] == alias]
    return team_filter



# EXAMPLE RUN:
#       data = getTeamScheduleById(key, "583ec70e-fb46-11e1-82cb-f4ce4684ea4c", 2021)
# DATA EXAMPLE:
#       sample_data/team_schedule_sample.json.json
def getTeamScheduleById(game_data, teamId):
    
    team_filter = [x for x in game_data if x["home"]["id"] == teamId or x["away"]["id"] == teamId]
    return team_filter


# EXAMPLE RUN:
#       data = getTeamGamesPriorToGame(key, "583ec70e-fb46-11e1-82cb-f4ce4684ea4c", 2021)
# DATA EXAMPLE:
#       sample_data/team_schedule_sample.json.json
def getTeamGamesPriorToGame(apikey, game_data, alias, year, enddate):
    date_filter = []
    for x in game_data:
        if parser.parse(x["scheduled"]) < enddate:
            date_filter.append(x)
    return date_filter

def getTeamGameIdsPriorToGame(game_data, enddate):
    date_filter = []
    for x in game_data:
        if x["status"] == "closed":
            if parser.parse(x["scheduled"]) < enddate:
                date_filter.append(x["id"])
    return date_filter

def getAllTeamIds(game_data):
    teams = []
    for game in game_data[0:200]:
        teams.append(game["home"]["id"])
    return set(teams)




def getGamesWithPriorArrByYear(apikey, year):
    schedule = getScheduleForYear(apikey, year)
    teams = getAllTeamIds(schedule)

    team_schedules = {}

    for team in teams:
        x = getTeamScheduleById(schedule, team)
        team_schedules[team] = x
        print(team)

    games = []

    for schedule_game in schedule:
        home_id = schedule_game["home"]["id"]
        away_id = schedule_game["away"]["id"]
        if schedule_game["status"] == "closed" and home_id in teams:
            game_date = parser.parse(schedule_game["scheduled"])
            print(game_date)
            home_prior_games = getTeamGameIdsPriorToGame(team_schedules[home_id], game_date)
            away_prior_games = getTeamGameIdsPriorToGame(team_schedules[away_id], game_date)
            home_win = 0
            away_win = 0
            if schedule_game["home_points"] > schedule_game["away_points"]:
                home_win = 1
            else:
                away_win = 1
            g = {"game_id": schedule_game["id"],"home_id": home_id, "away_id": away_id, "game_date": schedule_game["scheduled"], "home_alias": schedule_game["home"]["alias"], "away_alias": schedule_game["away"]["alias"], "home_points": schedule_game["home_points"], "away_points": schedule_game["away_points"], "home_win": home_win, "away_win": away_win, "home_prior_games": home_prior_games, "away_prior_games": away_prior_games}
            games.append(g)

    return games


def saveDataForYears(apikey, startyear, endyear):
    for x in range(startyear, endyear):
        data = getGamesWithPriorArrByYear(apikey, x)

        with open("data/reference_data_"+str(x)+".json", "w") as outfile:
            json.dump(data, outfile)
        
        print(str(x)+" Season Data Done!")



def saveGameStatData(apikeys, year):
    f = open("data/reference_data_"+str(year)+".json")
    data = json.load(f)
    games = []
    count = 0
    for idx, game in enumerate(data):
        print("************** RETRIEVING GAME **************")
        print("Index = "+ str(idx))
        print("Game ID = "+ game["game_id"])
        count +=1
        if count < 500:
            game_data = getGameDataById(apikeys[0], game["game_id"], 1)
        else:
            game_data = getGameDataById(apikeys[1], game["game_id"], 2)
        games.append(game_data)
        print(json.dumps(game_data)[:15])
        print("*********GAME SUCCESSFULLY ADDED************")
        time.sleep(1)
    with open("data/game_data_"+str(year)+".json", "w") as outfile:
        json.dump(games, outfile)


#keys = ["pdcu3z6b7tgm43qd9nwgx2jj", "cvwszfw53gg9bq42dyzqy7qs"]
#saveGameStatData(keys, 2013)




# MultiThreaded Approach
def getGameStatDataThreads(apikey, game_ids, conn):
    games = []
    for idx, game in enumerate(game_ids):
        print("Thread "+str(conn)+": ************** RETRIEVING GAME ON THREAD "+str(conn)+"**************")
        game_data = getGameDataById(apikey, game, conn)
        games.append(game_data)
        print("Thread "+str(conn)+": *********GAME SUCCESSFULLY "+ str(idx)+" ADDED************")
        time.sleep(1)
    return games


def saveGameStatDataThreads(apikeys, year):
    f = open("data/reference_data_"+str(year)+".json")
    data = json.load(f)
    game_ids = []
    for game in data:
        game_ids.append(game["game_id"])
    
    id1 = game_ids[:len(game_ids)//5]
    id2 = game_ids[len(game_ids)//5:2*len(game_ids)//5]
    id3 = game_ids[2*len(game_ids)//5:3*len(game_ids)//5]
    id4 = game_ids[3*len(game_ids)//5:4*len(game_ids)//5]
    id5 = game_ids[4*len(game_ids)//5:]
    que = queue.Queue()

    t1 = threading.Thread(target=lambda q: q.put(getGameStatDataThreads(apikeys[0], id1, 1)), args=(que, ))
    t2 = threading.Thread(target=lambda q: q.put(getGameStatDataThreads(apikeys[1], id2, 2)), args=(que, ))
    t3 = threading.Thread(target=lambda q: q.put(getGameStatDataThreads(apikeys[2], id3, 3)), args=(que, ))
    t4 = threading.Thread(target=lambda q: q.put(getGameStatDataThreads(apikeys[3], id4, 4)), args=(que, ))
    t5 = threading.Thread(target=lambda q: q.put(getGameStatDataThreads(apikeys[4], id5, 5)), args=(que, ))

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    games = []
    while not que.empty():
        r = que.get()
        games += r

    with open("data/game_data_"+str(year)+".json", "w") as outfile:
        json.dump(games, outfile)


keys = ["ust7hqkqb26d6avmvymhw2vg", "u79f6nvcju8azn8n4z3f96kv", "fnx642shubwxmx4cr7r5tv5j", "52mnfctgewhawnhh6hdhdm38", "4kghxfs3cmbg45g56t5aysk9"]
keys_2 = ["r6fpgt6czmjwynaqzfdxqyu4", "2vpvc3gqbcpnqef8zeutrys7", "zn8rhde22mgbzba5te3m23d7", "ttb5gcgzxreskkwmcwr6w8yb", "64dpv493dg7d7bvtpznv7t6d"]
#saveGameStatDataThreads(keys_2, 2020)

#saveDataForYears(keys_2[0], 2020, 2021)

print(json.dumps(getGameDataById(keys_2[1], "6be87e7c-ad7a-4361-bd37-535b411396a3", 1)))