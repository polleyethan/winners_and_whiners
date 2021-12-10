from os import lseek
import data_extractor as Parser


if __name__ == "__main__":
    player_actions = ["Shooter", "Assister", "Blocker", "Fouler", "Fouled", 
                    "Rebounder", "ViolationPlayer", "EnterGame", "LeaveGame", 
                    "TurnoverPlayer", "JumpballAwayPlayer", "JumpballHomePlayer"]
    filenames = ["../data/pbp-files/NBA_PBP_2015-16.csv", "../data/pbp-files/NBA_PBP_2016-17.csv",
                 "../data/pbp-files/NBA_PBP_2017-18.csv", "../data/pbp-files/NBA_PBP_2018-19.csv",
                "../data/pbp-files/NBA_PBP_2019-20.csv", "../data/pbp-files/NBA_PBP_2020-21.csv"]
    abbreviations = ["ATL", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC",
                    "LAL", "MEM","MIA", "MIL", "MIN", "NOH", "NOP", "NYK", "BKN", "OKC", "ORL", "PHI", 
                    "PHO", "POR", "SAC", "TOR", "UTH", "WAS"]
    #Parser.parse(filenames, player_actions)
    Parser.create_teams_csv(abbreviations)

