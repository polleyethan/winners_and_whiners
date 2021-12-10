import pandas as pd
import glob 
import numpy as np
from datetime import datetime
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import Sequence




"""
Class: DataSetCreator
Purpose: This class allows us to create dataframes for our team and player stats, which 
we will embed into a play-by-play model to predict live-time winning metrics. 
"""
class DataSetCreator():
    """
    Constructor 
    
    Parameters:
      Weights: This represents weights of seasons
      Team_remove_cols: Features of team data to disregard
      Player_remove_cols: Features of player data to disregard 
    """
    def __init__(self, normalized_data=True, weights=None, team_remove_cols=None, player_remove_cols=None, team_seasons=None, player_seasons=None):
        self.normalize = normalized_data
        if weights is None:
            self.weights = [0.005, 0.005, 0.005, 0.005, 0.005, 
                0.005, 0.005, 0.005, 0.005, 0.005,    
                0.01, 0.01, 0.01, 0.03, 0.03,   
                0.05, 0.10, 0.16, 0.25, 0.3]
        else:
            self.weights = weights
        if team_remove_cols is None: 
            self.team_remove_cols = ["abbreviation", "minutes_played", "name", "rank", "year", "field_goal_percentage", "free_throw_percentage", "opp_field_goal_percentage", "opp_free_throw_percentage", "opp_three_point_field_goal_percentage", 
                "opp_two_point_field_goal_percentage", "three_point_field_goal_percentage", "two_point_field_goal_percentage"]
        else:
            self.team_remove_cols = team_remove_cols
        if player_remove_cols is None:
            self.player_remove_cols = ["height", "nationality", "player_id", "team_abbreviation","salary","weight", "half_court_heaves", "half_court_heaves_made", "lost_ball_turnovers",
                    "net_plus_minus", "offensive_fouls", "other_turnovers", "passing_turnovers", "percentage_field_goals_as_dunks",
                  "percentage_of_three_pointers_from_corner", "percentage_shots_three_pointers", "percentage_shots_two_pointers", "point_guard_percentage",
                    "points_generated_by_assists", "power_forward_percentage", "shooting_fouls", "shooting_fouls_drawn", "shooting_guard_percentage", "shots_blocked",
                    "small_forward_percentage", "three_point_shot_percentage_from_corner", "three_pointers_assisted_percentage", "two_pointers_assisted_percentage",
                      "and_ones", "blocking_fouls", "center_percentage", ]
        else:
            self.player_remove_cols = player_remove_cols
        self.player_feat_len = 88 - len(self.player_remove_cols)
        self.team_feat_len = 48 - len(self.team_remove_cols)
        if team_seasons is None:
            self.team_seasons = [entry for entry in range(2019, 2000, -1)]
        else:
            self.team_seasons = team_seasons
        if player_seasons is None:
            self.player_seasons = [str(entry) + "-" + str(entry + 1)[-2:] for entry in range(2017, 1999, -1)]
        else:
            self.player_seasons = player_seasons 
        self.team_path = "teams"
        self.player_path = "players"
    
    def get_data_feature_size(self):
        return self.player_feat_len, self.team_feat_len
    """
    Input: None 
    Output: Using our self parameters, runs all the functions within that we need to 
    create and return player, team and normalized data. 
    """
    def process_data(self):
        team_df = self.build_teams(self.team_remove_cols, self.team_path, self.team_seasons)
        print("Built team dataframe")
        player_df = self.build_players(self.player_remove_cols, self.player_path, self.player_seasons)
        print("Built player data")
        team_names, player_names, norm_team_data, norm_player_data = self.normalize_data(team_df, player_df, self.normalize)
        print("Normalized data")
        return team_names, player_names, norm_team_data, norm_player_data
    """"
    build_teams: Function to build a team dataframe.
    Input: 
        remove_cols: list   columns to remove from the team data
        path: str,  format of the file path to the team csv's
        years: list[int]: list of years to consider seasons for 
    Output: A pandas dataframe containing aggregate stats for each team over the past 20 years,
            weighted by recent seasons.
    Purpose: Get a dataframe of average team statistics over the past 20 years. 
    """
    def build_teams(self, remove_cols: list, path: str, years: list) -> pd.DataFrame:
        teams = []
        # weights in the future can be potentially learned! 
        for csv_name in glob.glob("../data/%s/*.csv"%path):
            curr_df = pd.read_csv(csv_name)
            curr_df = curr_df.fillna(0)
            curr_df = curr_df[curr_df["year"].isin(years)]
            if curr_df.empty:
                continue
            else:
                #drop name columns if it isn't specified 
                #curr_df = curr_df.drop(columns=["abbreviation", "minutes_played", "name", "rank", "year"])
                curr_df = curr_df.drop(columns=remove_cols)
                summ_df = pd.DataFrame({})
                #remove weird-data point
                curr_cols = list(set(curr_df.columns) - set(["Unnamed: 0"]))
                curr_df = curr_df[curr_cols]

                #we are weighing by recency of season for stats
                for index, row in curr_df.iterrows():
                    curr_df.loc[index] = curr_df.loc[index] * self.weights[index]
                summ_df = pd.DataFrame(curr_df[curr_cols].sum(axis=0).to_numpy().reshape((1, -1)), columns=curr_cols)
                summ_df["abbreviation"] = [csv_name]
                summ_df.set_index("abbreviation")
                teams.append(summ_df)
        return pd.concat(teams)
    """
    This function takes our data frames and normalizes our data 
    Input:
      team_df: Team data
      player_df: Player data
    Output:
      team_names: Dictionary of teams
      player_names: Dictionary of players
      norm_team_data: Team data normalized to scale features
      norm_player_data: Player data normalized to scale features 
    """
    def normalize_data(self, team_df: pd.DataFrame, player_df: pd.DataFrame, normalize_data=True): 
        team_names = team_df["abbreviation"]
        player_names = player_df["player"]

        team_names = {entry.split("/")[-1].split(".")[0]:i for i,entry in enumerate(team_names)}
        player_names = {entry.split("/")[-1].split(".")[0]:i for i,entry in enumerate(player_names)}

        player_names['None'] = len(player_names)

        team_data = team_df.to_numpy()[:,:-1]
        player_data = player_df.to_numpy()[:,:-1]
        if normalize_data is True:
            #We want to scale the data for all our features 
            scaler = MinMaxScaler()
            scaler.fit(team_data)
            norm_team_data = scaler.transform(team_data)

            #We want to scale the data for all our features and normalize it 
            scaler = MinMaxScaler()
            scaler.fit(player_data)
            norm_player_data = scaler.transform(player_data)

            #print("normalized", norm_player_data)
            #norm_player_data = np.append(norm_player_data,np.zeros([1,len(player_names)]), axis=0)

            #could experiment with a normalization layer
        
            return team_names, player_names, norm_team_data, norm_player_data
        else:
            return team_names, player_names, team_data, player_data
    """
    build_players: Return dataframe of all players statistics aggregated over all their seasons. 

    Input: 
     remove_cols: list   columns to remove from the team data
     path: str,  format of the file path to the team csv's
     years: list[int]: list of years to consider seasons for 

    Output: 
        A pandas dataframe containing aggregate stats for each player over the past 20 years,
    Purpose: Get a dataframe of average player statistics over the past 20 years. 
    """
    def build_players(self, remove_columns: list, path: str, years: list) -> pd.DataFrame:
        players = []
        names = []
        pos_idx = {"PG": 0 , "SG" : 1, "SF" : 2, "PF" : 3, "C" : 4}
        # in the future try weighted average weighing recent seasons more
        for csv_name in glob.glob("../data/%s/*.csv"%path)[:]:
            curr_df = pd.read_csv(csv_name)
            curr_df = curr_df.fillna(0)
            curr_df = curr_df[curr_df["Unnamed: 0"].isin(years)]
            if curr_df.empty:
                continue
            else:
                names.append(csv_name)
                #curr_df = curr_df.drop(columns=["height", "nationality", "player_id", "team_abbreviation","salary","weight"])
                curr_df = curr_df.drop(columns=remove_columns)
                summed_df = pd.DataFrame({})

                curr_cols = list(set(curr_df.columns) - set(["Unnamed: 0", "position"]))
    
                summed_df = pd.DataFrame(curr_df[curr_cols].mean(axis=0).to_numpy().reshape((1, -1)), columns=curr_cols)

                summed_df["player"] = [csv_name]
                summed_df.set_index("player")
      
                players.append(summed_df)
        return pd.concat(players)



class PBP_Data_Baseline():
    def __init__(self, team_names: np.array, player_names:np.array, team_data:np.array, player_data:np.array, path: str, seasons: list, label='winner', team_feat=35, player_feat=57): 
        print("initialized trainable data")
        self.team_names = team_names
        self.player_names = player_names
        self.team_data = team_data
        self.player_data = player_data
        #self.total_features = 215
        self.team_features = team_feat
        self.player_features = player_feat
        self.total_features = self.team_features+3*self.player_features+8+1
        #print("baseline", self.team_features+3*self.player_features+8+1)
        self.label = label

        df_list = []
        for season in seasons:
            file_name = "../data/{path}/NBA_PBP_{season}.csv".format(path=path, season=season)
            curr_df = pd.read_csv(file_name)
            if curr_df.empty:
                continue
            df_list.append(curr_df)

        concat = [*df_list]
        self.df = pd.concat(concat)
        self.df["timestamp"] = [datetime.strptime("%s %s"%(date, time), "%B %d %Y %I:%M %p").timestamp() for date, time in self.df[["Date", "Time"]].to_numpy()]
        self.df.sort_values("timestamp")
        
        self.URL_ids = {i:url for i,url in enumerate(np.unique(self.df['URL'].to_numpy()))}
        self.id_URL = {url:i for i,url in enumerate(self.URL_ids.values())}
        self.df["game_ids"] = list(np.array([self.id_URL[url] for url in self.df["URL"]]))

        all_play_ids = []
        play_id = 0
        old_game_id = -1
        for game_id in self.df["game_ids"]:
            if game_id != old_game_id:
                play_id = 0
            all_play_ids.append(play_id)
            play_id += 1
            old_game_id = game_id
 
        self.df["play_ids"] = np.array(all_play_ids)
        self.max_plays = max(self.df["URL"].value_counts().to_numpy())
        self.num_games = len(self.df["URL"].value_counts().to_numpy())
        self.max_score = np.max(np.abs(self.df["HomeScore"] - self.df["AwayScore"]))
    def retrieve_team(self, name_lst):
        if len(name_lst) == 0:
            return np.zeros((0, self.team_features))
        return np.array([self.team_data[self.team_names[entry]] for entry in name_lst])

    def retrieve_player(self, name_lst):   
        if len(name_lst) == 0:
            return np.zeros((0, self.player_features))
        arr = []
        for entry in name_lst:
            try:
                data = self.player_data[self.player_names[entry.split("-")[-1].strip()]]
                arr.append(data)
            except:
                continue
        #val = np.array([self.player_data[self.player_names[entry.split("-")[-1].strip()]] for entry in name_lst[:-1]])
        #print("val1", val)
        val2 = np.array(arr)
        #print("val2", val2)
        return val2
        #return np.array([self.player_data[self.player_names[entry.split("-")[-1].strip()]] for entry in name_lst[:-1]])
        #return np.concatenate((val, np.array([self.player_data[self.player_names[entry2.split("-")[-1].strip()]])))

    def retrieve_team_2(self, name_lst: list):
        if len(name_lst) == 0:
            return np.zeros((0, self.team_features))
        team_name_list = []
        for entry in name_lst:
            try:
                key = self.team_names[entry]
                team_name_list.append(entry)
            except:
                continue
        return np.array([self.team_data[self.team_names[entry]] for entry in team_name_list])

    def retrieve_player_2(self, name_lst: list):

        if len(name_lst) == 0:
            return np.zeros((0, self.player_feat))
        player_name_list = []
        for entry in name_lst:
            try:
                key = self.player_names[entry.split("-")[-1].strip()]
                player_name_list.append(entry)
            except:
                continue
        return np.array([self.player_data[self.player_names[entry.split("-")[-1].strip()]] for entry in player_name_list]) 
    
    def retrieve_player_one(self, player_name_lst: list): 
        if len(player_name_lst) == 0:
            return np.zeros((0, self.player_features))

        arr = np.zeros((self.player_features))
        for entry in player_name_lst:
            try:
                arr_sub = np.array(self.player_data[self.player_names[entry.split("-")[-1].strip()]])
            except KeyError:
                arr_sub = np.zeros((self.player_features))
            arr = np.concatenate((arr, arr_sub), axis=0)
        return arr 
        #return np.array([self.player_data[self.player_names[entry.split("-")[-1].strip()]] for entry in player_name_lst]) 
       
    def retrieve_team_one(self, team_name_lst: list):
        if len(team_name_lst) == 0:
            return np.zeros((0, self.team_features))
        arr = np.zeros((self.team_features))
        for entry in team_name_lst:
            try:
                arr_sub = np.array(self.team_data[self.team_names[entry]])
            except KeyError:
                arr_sub = np.zeros((self.team_features))
            arr = np.concatenate((arr, arr_sub), axis=0)
        return arr 
        #return np.array([self.team_data[self.team_names[entry]] for entry in team_name_lst])
    
    def split_data(self, split_type: str, split_frac: int = 0.9):
        train_size = int(self.num_games*split_frac)
        test_size = self.num_games - train_size
        if split_type == "train":
            return np.arange(train_size)
        else:
            return np.arange(train_size, self.num_games)
    def get_entries(self, idxs):
        df = self.df[self.df["game_ids"].isin(idxs)]

        reverse_dict = {idx:i for i,idx in enumerate(idxs)}

        num_games = len(idxs)
        max_plays = self.max_plays
        total_features = self.total_features
        team_feat = self.team_features
        player_feat = self.player_features

        game_feats = np.zeros((num_games, self.max_plays, self.total_features))

        game_ids = df["game_ids"]
        game_ids = np.array([reverse_dict[entry] for entry in game_ids])
        all_play_ids = df["play_ids"]

        game_ids = game_ids.astype(int)
        all_play_ids = all_play_ids.astype(int)

        # print("game_ids", game_ids)
        # print("ids", all_play_ids)
        # print(print(df["AwayPlay"]==df["AwayPlay"]))


        team_data = self.retrieve_team(df["AwayTeam"][df["AwayPlay"] == df["AwayPlay"]])
        game_feats[game_ids[ df["AwayPlay"] == df["AwayPlay"]], all_play_ids[df["AwayPlay"] == df["AwayPlay"]], :team_feat] = team_data
        # print("c.shape", c.shape)
        # print("team_data", team_data.shape)
        

        team_data = self.retrieve_team(df["HomeTeam"][df["HomePlay"] == df["HomePlay"]])
        d = game_feats[game_ids[ df["HomePlay"] == df["HomePlay"]], all_play_ids[df["HomePlay"] == df["HomePlay"]], :team_feat]
        game_feats[game_ids[ df["HomePlay"] == df["HomePlay"]], all_play_ids[df["HomePlay"] == df["HomePlay"]], :team_feat] = team_data
        # print("team_data_2", team_data.shape)
        # print("d", d.shape)


        player_data = self.retrieve_player(df["Shooter"][df["Shooter"] == df["Shooter"]])
        
        e = game_feats[game_ids[df["Shooter"] == df["Shooter"]], all_play_ids[df["Shooter"] == df["Shooter"]], team_feat:team_feat+player_feat]
        game_feats[game_ids[df["Shooter"] == df["Shooter"]], all_play_ids[df["Shooter"] == df["Shooter"]], team_feat:team_feat+player_feat] = player_data

        # print("player_data_1", player_data.shape)
        # print("e.shape", e.shape)
        

        player_data = self.retrieve_player(df["Blocker"][df["Blocker"] == df["Blocker"]])
        game_feats[game_ids[df["Blocker"] == df["Blocker"]], all_play_ids[df["Blocker"] == df["Blocker"]], team_feat+player_feat: team_feat+2*player_feat] = player_data

        player_data = self.retrieve_player(df["Assister"][df["Assister"] == df["Assister"]])
        game_feats[game_ids[df["Assister"] == df["Assister"]], all_play_ids[df["Assister"] == df["Assister"]], team_feat+2*player_feat:team_feat+3*player_feat] = player_data

        player_data = self.retrieve_player(df["FreeThrowShooter"][df["FreeThrowShooter"] == df["FreeThrowShooter"]])
        game_feats[game_ids[df["FreeThrowShooter"] == df["FreeThrowShooter"]], all_play_ids[df["FreeThrowShooter"] == df["FreeThrowShooter"]], team_feat:team_feat+player_feat] = player_data

        player_data = df["Rebounder"][df["Rebounder"] == df["Rebounder"]]
        player_data[player_data == "Team"] = "None"
        player_data = self.retrieve_player(player_data)
        #player_data = self.to_shape(player_data, )
        d = game_feats[game_ids[df["Rebounder"] == df["Rebounder"]], all_play_ids[df["Rebounder"] == df["Rebounder"]], team_feat:team_feat+player_feat]
        player_data = self.shape_array(player_data, d)

        game_feats[game_ids[df["Rebounder"] == df["Rebounder"]], all_play_ids[df["Rebounder"] == df["Rebounder"]], team_feat:team_feat+player_feat] = player_data
        
        player_data = self.retrieve_player(df["Fouler"][df["Fouler"] == df["Fouler"]])
        game_feats[game_ids[df["Fouler"] == df["Fouler"]], all_play_ids[df["Fouler"] == df["Fouler"]], team_feat:team_feat+player_feat] = player_data

        player_data = self.retrieve_player(df["Fouled"][df["Fouled"] == df["Fouled"]])
        game_feats[game_ids[df["Fouled"] == df["Fouled"]], all_play_ids[df["Fouled"] == df["Fouled"]], team_feat+player_feat: team_feat+2*player_feat] = player_data

        player_data = self.retrieve_player(df["EnterGame"][df["EnterGame"] == df["EnterGame"]])
        game_feats[game_ids[df["EnterGame"] == df["EnterGame"]], all_play_ids[df["EnterGame"] == df["EnterGame"]], team_feat:team_feat+player_feat] = player_data

        player_data = self.retrieve_player(df["LeaveGame"][df["LeaveGame"] == df["LeaveGame"]])
        game_feats[game_ids[df["LeaveGame"] == df["LeaveGame"]], all_play_ids[df["LeaveGame"] == df["LeaveGame"]], team_feat+2*player_feat:team_feat+3*player_feat] = player_data


        game_feats[game_ids, all_play_ids, team_feat+3*player_feat] = df["Quarter"]/5
        game_feats[game_ids, all_play_ids, team_feat+3*player_feat+1] = df["SecLeft"]/720
        game_feats[game_ids[df["ShotType"] == df["ShotType"]], all_play_ids[df["ShotType"] == df["ShotType"]], team_feat+3*player_feat+2] = np.array([int(entry[0]) for entry in df["ShotType"][df["ShotType"] == df["ShotType"]].to_numpy()])/3.0
        game_feats[game_ids[df["FreeThrowShooter"] == df["FreeThrowShooter"]], all_play_ids[df["FreeThrowShooter"] == df["FreeThrowShooter"]], team_feat+3*player_feat+2] = 2/3.0
        game_feats[game_ids[df['ReboundType'] == "defensive"], all_play_ids[df['ReboundType'] == "defensive"], team_feat+3*player_feat+2] = 1/3.0

        game_feats[game_ids[df["ShotOutcome"] == "make"], all_play_ids[df["ShotOutcome"] == "make"], team_feat+3*player_feat+2] = game_feats[game_ids[df["ShotOutcome"] == "make"], all_play_ids[df["ShotOutcome"] == "make"], team_feat+3*player_feat+2]



        game_feats[game_ids[df["Shooter"] == df["Shooter"]], all_play_ids[df["Shooter"] == df["Shooter"]], team_feat+3*player_feat+3] = 1
        game_feats[game_ids[df["FreeThrowShooter"] == df["FreeThrowShooter"]], all_play_ids[df["FreeThrowShooter"] == df["FreeThrowShooter"]], team_feat+3*player_feat+4] = 1
        game_feats[game_ids[df["Rebounder"] == df["Rebounder"]], all_play_ids[df["Rebounder"] == df["Rebounder"]], team_feat+3*player_feat+5] = 1
        game_feats[game_ids[df["Fouler"] == df["Fouler"]], all_play_ids[df["Fouler"] == df["Fouler"]], team_feat+3*player_feat+6] = 1
        game_feats[game_ids[df["EnterGame"] == df["EnterGame"]], all_play_ids[df["EnterGame"] == df["EnterGame"]], team_feat+3*player_feat+7] = 1
        game_feats[game_ids[df["TimeoutTeam"] == df["TimeoutTeam"]], all_play_ids[df["TimeoutTeam"] == df["TimeoutTeam"]], team_feat+3*player_feat+8] = 1
    
        # print("score-home", df["HomeScore"])
        # print("score-away", df["AwayScore"])


        game_feats[game_ids, all_play_ids, -1] = df["HomeScore"]
        # print("game_ids", game_ids)
        # print("play_ids", all_play_ids)
        # print("game_feats", game_feats[game_ids, all_play_ids, -1])
        max_home = np.max(game_feats[:,:,-1], axis=1)
        game_feats[:, :, -1] *= 0
        game_feats[game_ids, all_play_ids, -1] = df["AwayScore"]
        max_away = np.max(game_feats[:,:,-1], axis=1)
        game_feats[:, :, -1] *= 0


        game_feats[:, :, -1] = ( (max_home-max_away).reshape((-1, 1)) * np.ones((num_games, max_plays)) )
        game_feats[game_ids, all_play_ids, -1] = df["HomeScore"]-df["AwayScore"]
        y_true_line = np.array(game_feats[game_ids, all_play_ids, -1])
        game_feats[:,:,-2] = (game_feats[:,:,-1] > 0).astype(int)
        game_feats[:,:, -1] = np.abs(game_feats[:,:,-1])
        game_feats[:,:,-1] = game_feats[:,:,-1]/self.max_score

        games = game_feats
        label_type = self.label
        if label_type == "winner":
            y_true = []
            winner_check = np.zeros((num_games, max_plays))
            winner_check[game_ids, all_play_ids] = df["WinningTeam"] == df["HomeTeam"]
            winner_check = winner_check[:, 0]
            y_true = np.array([[0,1] if entry else [1,0] for entry in winner_check])
        else:
            winner_check = np.zeros((num_games, max_plays))
            for game_id in game_ids:
                print("game_id", game_id)
        return games, y_true
    def shape_array(self, a, b, axis=0):
        shape = np.shape(a)
        padded_array_shape = np.shape(b)
        padded_array = np.zeros(padded_array_shape)
        padded_array[:shape[0], :shape[1]] = a 
        #print("shape of pad", padded_array.shape)
        return padded_array

    def to_shape(a, shape):
        y_, x_ = shape
        y, x = a.shape
        y_pad = (y_-y)
        x_pad = (x_-x)
        return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant')

class Traininable_PBPData(Sequence):
    def __init__(self, pbp_data: PBP_Data_Baseline, data_type='Train', split=0.9, batch_size=64, play_to_grab=150):
        self.pbp_data = pbp_data
        self.idxs = self.pbp_data.split_data(data_type, split)
        self.size = len(self.idxs)
        self.batch_size = batch_size
        self.play_window = play_to_grab
        self.curr_idxs = self.idxs.copy()
        print("Created PBPDatabase")
    def __len__(self):
        return math.ceil(self.size / self.batch_size)
    def __getitem__(self, idx):
       if len(self.curr_idxs) == 0:
          self.curr_idxs = self.idxs.copy()
       np.random.shuffle(self.curr_idxs)
       curr_idx = self.curr_idxs[:self.batch_size]
       self.curr_idxs = self.curr_idxs[self.batch_size:]
       data, gts = self.pbp_data.get_entries(curr_idx)
       return data[:, :self.play_window, :], gts
    def get_dataframe_for_testing(self):
       if len(self.curr_idxs) == 0:
            self.curr_idxs = self.idxs.copy()
            np.random.shuffle(self.curr_idxs)
       curr_idx = self.curr_idxs[:self.batch_size]
       self.curr_idxs = self.curr_idxs[self.batch_size:]
       data, gts = self.pbp_data.get_entries(curr_idx)
       return data[:, :self.play_window, :], gts

class Data_Modeler():
    def __init__(self, label='winner', normalized_data=True, weights=None, team_remove_cols=None, player_remove_cols=None, team_seasons=None, player_seasons=None,
                    split=0.9, batch_size=150, play_window=150, team_names=None, player_names=None,
                    team_data=None, player_data=None, baseline=None, PBPTrain=None, PBPTest=None):
        self.label = label
        self.team_names = team_names
        self.player_names = player_names
        self.team_data = team_data
        self.player_data = player_data
        self.baseline = baseline
        self.PBPTrainData = PBPTrain
        self.PBPTestData = PBPTest 
        self.normalized_data = normalized_data
        self.weights = weights
        self.team_remove_cols = team_remove_cols
        self.player_remove_cols = player_remove_cols
        self.player_seasons = player_seasons
        self.team_seasons = team_seasons
        self.split = split 
        self.batch_size = batch_size
        self.play_window = play_window
        if self.player_seasons is None:
            self.player_seasons = [str(entry) + "-" + str(entry + 1)[-2:] for entry in range(2017, 1999, -1)]
        self.season_files = [str(entry) + "-" + str(entry + 1)[-2:] for entry in range(2017, 2014, -1)]
        self.ds_creator = DataSetCreator(normalized_data=self.normalized_data, weights=self.weights, team_remove_cols=self.team_remove_cols, player_remove_cols=self.player_remove_cols, team_seasons=self.team_seasons, player_seasons=self.player_seasons)
        
    def return_team_names(self):
        return self.team_names 
    def get_normalized_data(self):
        if self.team_names is None or self.player_names is None or self.team_data is None or self.player_data is None:
                self.team_names, self.player_names, self.team_data, self.player_data = self.ds_creator.process_data()
        return self.team_names, self.player_names, self.team_data, self.player_data 
   
    def getPBP_Base(self):
        if self.team_names is None or self.player_names is None or self.team_data is None or self.player_data is None:
                self.team_names, self.player_names, self.team_data, self.player_data = self.ds_creator.process_data()
        if self.baseline is None: 
                self.baseline = PBP_Data_Baseline(self.team_names, self.player_names, self.team_data, self.player_data, "games", self.season_files, label=self.label)
        return self.baseline 
    
    def get_PBPData(self):
        print("getting training and testing data")
        if self.PBPTrainData is None:
            if self.team_names is None or self.player_names is None or self.team_data is None or self.player_data is None:
                self.team_names, self.player_names, self.team_data, self.player_data = self.ds_creator.process_data()
            if self.baseline is None: 
                self.baseline = PBP_Data_Baseline(self.team_names, self.player_names, self.team_data, self.player_data, "games", self.season_files, label=self.label)
            self.PBPTrainData = Traininable_PBPData(self.baseline, data_type='Train', split=self.split, batch_size=self.batch_size, play_to_grab=self.play_window)
        if self.PBPTestData is None:
            if self.team_names is None or self.player_names is None or self.team_data is None or self.player_data is None:
                self.team_names, self.player_names, self.team_data, self.player_data = self.ds_creator.process_data()
            if self.baseline is None: 
                self.baseline = PBP_Data_Baseline(self.team_names, self.player_names, self.team_data, self.player_data, "games", self.season_files, label=self.label)
            self.PBPTestData = Traininable_PBPData(self.baseline, data_type='Test', split=self.split, batch_size=self.batch_size, play_to_grab=self.play_window)
        return self.PBPTrainData, self.PBPTestData 
