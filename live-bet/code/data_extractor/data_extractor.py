import pandas as pd 
import numpy as np
import os 
import threading
import time
from sportsreference.nba.roster import Player, Roster
#from sportsreference.nba.teams import Teams
from sportsipy.nba.teams import Teams, Team


"""
This does all of our parsing action
Input:
  file_name: list of Strings representing csv-files to parse players for 
  criteria: List of actions that we will look at to find players 
Output:
  None
Purpose:
  Create player and team CSV's for us to use and index plays from later.
""" 
def parse(file_names: list, criteria:list):
    plist = get_list_of_players(file_names, criteria)
    print("STEP 1 DONE: Got list of players")
    create_player_csv(plist)
    print("STEP 2 DONE: Created list of players")
    create_teams_csv()
    print("STEP 3 DONE: Created list of teams")

"""
This method goes through all our historical
play-by-play data and returns a list of players
Input:
  file_name: list of Strings representing csv-files to parse players for 
  criteria: List of actions that we will look at to find players 
Output:
  output: list of unique players
"""
def get_list_of_players(file_names: list, criteria: list) -> list:
    output = []
    #get the dataframe associated with the season
    for name in file_names:
        df = pd.read_csv(name) 
        for criterion in criteria:
            output = np.concatenate([output, np.unique(df[criterion].dropna().to_numpy())])
    plist = list(np.unique(output))
    for i, row in enumerate(plist):
        plist[i] = plist[i].split(" ")[-1]
    plist = [i for i in plist if i[-1]!= 'c']
    return plist
"""
Input: 
    p_list: List of player ID's to search for actions
Output: 
    None 
Purpose: 
    To create csv's for player's actions and player data
"""
def create_player_csv(p_list: list) -> None:
    for p in p_list:
        try:
            curr_player = Player(p)
            csv_path = "../data/player-data/{player}.csv".format(player=str(curr_player.name))
            if not os.path.exists(csv_path):
                curr_player.dataframe.to_csv(csv_path)
                print("GOT:", curr_player.name)
            else:
                print("Player exists", str(p))
                continue
        except:
            continue
"""
Input: 
    None 
Output: 
    None
Purpose:
    To create a CSV for each team in team-data
"""
def create_teams_csv(team_abbrv: list):
    seasons = [str(entry) for entry in range(2019, 1999, -1)]
    #teams = Team(team_name="NYK")
    for abbr in team_abbrv:
        team_df = []
        for season in seasons:
            try:
                curr_df = Teams(year=season)[abbr].dataframe
                print("curr_df", curr_df)
            except:
                print("Issue with tracking dataframe")
                continue
            curr_df["year"] = season
            team_df.append(curr_df)
        team_df = pd.concat(team_df)
        csv_path = "../data/team-data/{team}.csv".format(team=str(abbr))
        if not os.path.exists(csv_path):
            team_df.to_csv(csv_path)
        else:
            print("Path exists for Team CSV")
    
    """"
    seasons = [str(entry) for entry in range(2019, 2010, -1)]
    teams = Teams()

    num_teams = len(Teams())
    print("num_teams", num_teams)
    team_set = []
    for season in seasons:
        team_set.append(Teams(year=season))

    print(team_set)

    team_abbrv = [[team.abbreviation for team in teams] for teams in team_set]
    team_abbrv = sum(team_abbrv, [])
    team_abbrv = np.unique(team_abbrv)
    print("team_abbrv", team_abbrv)

    for abbr in team_abbrv:
        team_df = []
        for season in seasons:
            try:
                curr_df = Teams(year=season)[abbr].dataframe
            except:
                print("Issue with tracking dataframe")
                continue
            curr_df["year"] = season
            team_df.append(curr_df)
        team_df = pd.concat(team_df)
        csv_path = "../data/team-data/{team}.csv".format(team=str(abbr))
        if not os.path.exists(csv_path):
            team_df.to_csv(csv_path)
        else:
            print("Path exists for Team CSV")
        """

