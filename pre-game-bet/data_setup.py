import json


data_fields_to_aggregate = [ 'field_goals_made', 'field_goals_att', 'three_points_made', 'three_points_att', 'two_points_made',
 'two_points_att', 'free_throws_made', 'free_throws_att', 'offensive_rebounds', 'defensive_rebounds',
  'rebounds', 'assists', 'turnovers', 'steals', 'blocks', 'points', 'fast_break_pts',
   'second_chance_pts', 'team_turnovers', 'points_off_turnovers', 'team_rebounds', 'points_in_paint']



def get_input_fields(game_arr, home_prev_games, away_prev_games, home_id, away_id):
    values = {}
    for field in data_fields_to_aggregate:
        values["home_"+field] = 0
        values["away_"+field] = 0
        values["home_"+field+"_against"] = 0
        values["away_"+field+"_against"] = 0

    #Sum Home Games Prev Values
    for game in home_prev_games:
        team = "away"
        opponent = "home"
        if game_arr[game]["home"]["id"] == home_id:
            team = "home"
            opponent = "away"

        for field in data_fields_to_aggregate:
            try:
                values["home_"+field] += game_arr[game][team]["statistics"][field]
                values["home_"+field+"_against"] += game_arr[game][opponent]["statistics"][field]
            except KeyError as e:
                print("Key Error on game "+game+": "+str(e))
                print(game_arr[game]["home"]["name"]+ " vs "+game_arr[game]["away"]["name"] + " on "+ game_arr[game]["scheduled"])
        
        



    # Sum Away Games Prev Values
    for game in away_prev_games:
        team = "away"
        opponent = "home"
        if game_arr[game]["home"]["id"] == away_id:
            team = "home"
            opponent = "away"
        for field in data_fields_to_aggregate:
            try:
                values["away_"+field] += game_arr[game][team]["statistics"][field]
                values["away_"+field+"_against"] += game_arr[game][opponent]["statistics"][field]
            except KeyError as e:
                print("Key Error on game "+game+": "+str(e))


    
    #Get Averages, Percentages, Ratios
    for team in ["home", "away"]:
        num_games = len(home_prev_games)

        if team == "away":
            num_games = len(away_prev_games)

        if num_games > 0:
            # Get Percentages and Ratios
            values[team+"_field_goals_pct"] = values[team+"_field_goals_made"] / values[team+"_field_goals_att"] if values[team+"_field_goals_att"] else 0
            values[team+"_three_points_pct"] = values[team+"_three_points_made"] / values[team+"_three_points_att"] if values[team+"_three_points_att"] else 0
            values[team+"_two_points_pct"] = values[team+"_two_points_made"] / values[team+"_two_points_att"] if values[team+"_two_points_att"] else 0
            values[team+"_free_throws_pct"] = values[team+"_free_throws_made"] / values[team+"_free_throws_att"] if values[team+"_free_throws_att"] else 0
            values[team+"_assists_turnover_ratio"] = values[team+"_assists"]/values[team+"_turnovers"] if values[team+"_turnovers"] else 0
            # Against
            values[team+"_field_goals_pct_against"] = values[team+"_field_goals_made_against"] / values[team+"_field_goals_att_against"] if values[team+"_field_goals_att_against"] else 0
            values[team+"_three_points_pct_against"] = values[team+"_three_points_made_against"] / values[team+"_three_points_att_against"] if values[team+"_three_points_att_against"] else 0
            values[team+"_two_points_pct_against"] = values[team+"_two_points_made_against"] / values[team+"_two_points_att_against"] if values[team+"_two_points_att_against"] else 0
            values[team+"_free_throws_pct_against"] = values[team+"_free_throws_made_against"] / values[team+"_free_throws_att_against"] if values[team+"_free_throws_att_against"] else 0
            values[team+"_assists_turnover_ratio_against"] = values[team+"_assists_against"]/values[team+"_turnovers_against"] if values[team+"_turnovers_against"] else 0

            for field in data_fields_to_aggregate:
                values[team+"_"+field] = values[team+"_"+field] / num_games if num_games else 0
                values[team+"_"+field+"_against"] = values[team+"_"+field+"_against"] / num_games if num_games else 0
    
    return values

def data_aggregator(year, post_17 = False):
    reference_file = open("data/reference_data_"+str(year)+".json")
    reference_data = json.load(reference_file)

    stats_file = open("data/game_data_"+str(year)+".json")
    stats_data = json.load(stats_file)

    stats_hash = {}

    for game in stats_data:
        stats_hash[game["id"]] = game
    
    game_aggregate_data = []
    for idx, ref_game in enumerate(reference_data):
        if len(ref_game["home_prior_games"]) > 0 and len(ref_game["home_prior_games"])  > 0:
            print("Game "+str(idx))
            game_data = {}
            game_data["reference"] = {
                "sportsradar_id": ref_game["game_id"],
                "game_date": ref_game["game_date"],
                "home_team": {
                    "sportsradar_teamid": ref_game["home_id"],
                    "team_abbrev": ref_game["home_alias"],
                    "points": ref_game["home_points"],
                    "win": ref_game["home_win"]
                },
                "away_team": {
                    "sportsradar_teamid": ref_game["away_id"],
                    "team_abbrev": ref_game["away_alias"],
                    "points": ref_game["away_points"],
                    "win": ref_game["away_win"]
                }
            }
            game_data["inputs"] = get_input_fields(stats_hash, ref_game["home_prior_games"], ref_game["away_prior_games"], ref_game["home_id"], ref_game["away_id"])
            game_data["ouputs"] = {
                "home_points": ref_game["home_points"],
                "away_points": ref_game["away_points"],
                "home_win": ref_game["home_win"],
                "away_win": ref_game["away_win"]
            }

            game_aggregate_data.append(game_data)

    return game_aggregate_data

for year in range(2013, 2021):
    da = data_aggregator(year)

    with open("data/composite/composite_"+str(year)+".json", "w") as out:
        json.dump(da, out)



def get_data(filename):
    pass


#data_aggregator(2020)

"""
    #Get Averages
    for team in ["home", "away"]:
        num_games = len(home_prev_games)

        if team == "away":
            num_games = len(away_prev_games)

        values[team+"_field_goals_pct"] = values[team+"_field_goals_made"] / values[team+"_field_goals_att"]
        values[team+"_field_goals_made"] = values[team+"_field_goals_made"]/num_games
        values[team+"_field_goals_att"] = values[team+"_field_goals_att"]/num_games

        values[team+"_three_points_pct"] = values[team+"_three_points_made"] / values[team+"_three_points_att"]
        values[team+"_three_points_made"] = values[team+"_three_points_made"]/num_games
        values[team+"_three_points_att"] = values[team+"_three_points_att"]/num_games

        values[team+"_two_points_pct"] = values[team+"_two_points_made"] / values[team+"_two_points_att"]
        values[team+"_two_points_made"] = values[team+"_two_points_made"]/num_games
        values[team+"_two_points_att"] = values[team+"_two_points_att"]/num_games

        values[team+"_free_throws_pct"] = values[team+"_free_throws_made"] / values[team+"_free_throws_att"]
        values[team+"_free_throws_made"] = values[team+"_free_throws_made"]/num_games
        values[team+"_free_throws_att"] = values[team+"_free_throws_att"]/num_games

        values[team+"_rebounds"] = values[team+"_rebounds"]/num_games
        values[team+"_offensive_rebounds"] = values[team+"_offensive_rebounds"]/num_games
        values[team+"_defensive_rebounds"] = values[team+"_defensive_rebounds"]/num_games
        values[team+"_steals"] = values[team+"_steals"]/num_games
        values[team+"_blocks"] = values[team+"_blocks"]/num_games
        values[team+"_blocked_att"] = values[team+"_blocked_att"]/num_games
        values[team+"_turnovers"] = values[team+"_turnovers"]/num_games
        values[team+"_points_off_turnovers"] = values[team+"_points_off_turnovers"]/num_games
        values[team+"_points_in_paint"] = values[team+"_points_in_paint"]/num_games
        values[team+"_second_chance_pts"] = values[team+"_second_chance_pts"]/num_games
        values[team+"_fast_break_pts"] = values[team+"_second_chance_points"]/num_games
        values[team+"_points"] = values[team+"_points"]/num_games
        values[team+"_team_rebounds"] = values[team+"_team_rebounds"]/num_games
        values[team+"_team_turnovers"] = values[team+"_team_turnovers"]/num_games
        values[team+"_assists_turnover_ratio"] = values[team+"_assists"]/values[team+"_turnovers"]



data_fields = [ 'field_goals_made', 'field_goals_att', 'field_goals_pct', 'three_points_made', 'three_points_att', 'three_points_pct', 'two_points_made',
 'two_points_att', 'two_points_pct', 'blocked_att', 'free_throws_made', 'free_throws_att', 'free_throws_pct', 'offensive_rebounds', 'defensive_rebounds',
  'rebounds', 'assists', 'turnovers', 'steals', 'blocks', 'assists_turnover_ratio', 'points', 'fast_break_pts',
   'second_chance_pts', 'team_turnovers', 'points_off_turnovers', 'team_rebounds', 'points_in_paint']
"""