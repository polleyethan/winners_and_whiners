import data_constructor as dc
import pandas as pd
import numpy as np 
import modelwrapper as mo

#TODO: Add experiments, make modifications and run these experiments potentially in parallel 
#Create separate files maybe but its up to you.

if __name__ == "__main__":
    """
    This main method contains comments to help run some of the modularized code. If you want
    to play around with the models, go to modelwrapper.py, create a new Model class in the same format
    and add it to the ModelWrapper() constructor. 
    """


    """
    Below, This is the data_modeler class which is used to extract our data from its format
    and turn our data into traininable test/train data. We can use custom parameters in the data
    modeler to change features of our data. 

    To feed the modeler parameters or change it you want to provide it with the variables in its name 

    These are the parameters you can edit to change what data goes into the model and play around with
      team_remove_cols: These are the features of the team data that you can remove. 
      player_remove_cols: These are the features of the play_data that you can remove 
      split = The train/test split fraction 
      batch_size = The size of our batches of plays (Default is 0.5)
      play_window: The window size of the plays we use to classify (Default is 150)
      team_seasons: These are the seasons of data we source for teams
      player_seasons: These are the seasons of data we source for teams 

    USAGE:

    To use default data_modelers to get our train and test_data 
      data_modeler = dc.Data_Modeler() 
    
    To supply an argument (all optional), take the name of the argument and add an extra value that 
    corresponds
      ex: data_modeler = dc.Data_Modeler(split=0.5, batch_size=300, play_window=200, player_remove_cols=[["Free Throw Percentage]])
    
    These are the columns u can remove for team or player data.

    player_data = ['and_ones' 'assist_percentage' 'assists' 'block_percentage'
    'blocking_fouls' 'blocks' 'box_plus_minus' 'center_percentage'
    'defensive_box_plus_minus' 'defensive_rebound_percentage'
    'defensive_rebounds' 'defensive_win_shares' 'dunks'
    'effective_field_goal_percentage' 'field_goal_attempts'
    'field_goal_perc_sixteen_foot_plus_two_pointers'
    'field_goal_perc_ten_to_sixteen_feet' 'field_goal_perc_three_to_ten_feet'
    'field_goal_perc_zero_to_three_feet' 'field_goal_percentage'
    'field_goals' 'free_throw_attempt_rate' 'free_throw_attempts'
    'free_throw_percentage' 'free_throws' 'games_played' 'games_started'
    'half_court_heaves' 'half_court_heaves_made' 'height'
    'lost_ball_turnovers' 'minutes_played' 'nationality' 'net_plus_minus'
    'offensive_box_plus_minus' 'offensive_fouls'
    'offensive_rebound_percentage' 'offensive_rebounds'
    'offensive_win_shares' 'on_court_plus_minus' 'other_turnovers'
    'passing_turnovers' 'percentage_field_goals_as_dunks'
    'percentage_of_three_pointers_from_corner'
    'percentage_shots_three_pointers' 'percentage_shots_two_pointers'
    'percentage_sixteen_foot_plus_two_pointers'
    'percentage_ten_to_sixteen_footers' 'percentage_three_to_ten_footers'
    'percentage_zero_to_three_footers' 'personal_fouls'
    'player_efficiency_rating' 'player_id' 'point_guard_percentage' 'points'
    'points_generated_by_assists' 'position' 'power_forward_percentage'
    'salary' 'shooting_distance' 'shooting_fouls' 'shooting_fouls_drawn'
    'shooting_guard_percentage' 'shots_blocked' 'small_forward_percentage'
    'steal_percentage' 'steals' 'take_fouls' 'team_abbreviation'
    'three_point_attempt_rate' 'three_point_attempts'
    'three_point_percentage' 'three_point_shot_percentage_from_corner'
    'three_pointers' 'three_pointers_assisted_percentage'
    'total_rebound_percentage' 'total_rebounds' 'true_shooting_percentage'
    'turnover_percentage' 'turnovers' 'two_point_attempts'
    'two_point_percentage' 'two_pointers' 'two_pointers_assisted_percentage'
    'usage_percentage' 'value_over_replacement_player' 'weight' 'win_shares'
    'win_shares_per_48_minutes']]
  
    team_data = ['abbreviation' 'assists' 'blocks' 'defensive_rebounds'
    'field_goal_attempts' 'field_goal_percentage' 'field_goals'
    'free_throw_attempts' 'free_throw_percentage' 'free_throws'
    'games_played' 'minutes_played' 'name' 'offensive_rebounds' 'opp_assists'
    'opp_blocks' 'opp_defensive_rebounds' 'opp_field_goal_attempts'
    'opp_field_goal_percentage' 'opp_field_goals' 'opp_free_throw_attempts'
    'opp_free_throw_percentage' 'opp_free_throws' 'opp_offensive_rebounds'
    'opp_personal_fouls' 'opp_points' 'opp_steals'
    'opp_three_point_field_goal_attempts'
    'opp_three_point_field_goal_percentage' 'opp_three_point_field_goals'
    'opp_total_rebounds' 'opp_turnovers' 'opp_two_point_field_goal_attempts'
    'opp_two_point_field_goal_percentage' 'opp_two_point_field_goals'
    'personal_fouls' 'points' 'rank' 'steals'
    'three_point_field_goal_attempts' 'three_point_field_goal_percentage'
    'three_point_field_goals' 'total_rebounds' 'turnovers'
    'two_point_field_goal_attempts' 'two_point_field_goal_percentage'
    'two_point_field_goals' 'year']

    To remove columns just feed the following args

    data_modeler = dc.Data_Modeler(team_data=['two_point_field_goal_attempts', 'year', ['turnovers])
    data_modeler = dc.Data_Modeler(player_data=['field_goals', 'steal_percentage'])
    """
    #get the datamodeler 
    data_modeler = dc.Data_Modeler() 

    #This returns our train and test data
    train_data, test_data = data_modeler.get_PBPData()
    
    """
    Our ModelWrapper class is a class that contains all of our models and simply runs one 
    of them by supplying an integer argument that corresponds to our model (check modelwrapper.py)

    To use it simply call model_wrapper = mo.ModelWrapper(num_model)

    Then to train/test (it should write to a file), call model_wrapper.run(train_data, test_data)
    An optional parameter for .run is epoch (you can change number it runs for): ex:

    model_wrapper.run(train_data, test_data, epoch=10)


    If you want to create your own model, reduce numbers of dense layers, play around with features 
    just copy one of the models, change some of its features and return it. 
    """
    #model_wrapper = mo.ModelWrapper(1)
    #model_wrapper.run(train_data, test_data, epochs=5)
    
    """
    #You can use this to run all the models 
    for i in range(1, 15, 1):
      try:
        model_wrapper = mo.ModelWrapper(i)
        model_wrapper.run(train_data, test_data, epochs=300)
      except:
        print("Error: Model", i, "failed to run, need to check call function")
        continue 
    
    """
    #Or try to run specific models here
    model_wrapper = mo.ModelWrapper(12)
    model_wrapper.run(train_data, test_data, epochs=300)
    
    
    
    
    
    