import os

class Score():
    # Ref: https://tetris.wiki/Scoring
    score_table = {
        "single": 100,
        "double": 300,
        "triple": 500,
        "tetris": 800,
        "t-spin mini no lines": 100,
        "t-spin no lines": 400,
        "t-spin mini single": 200,
        "t-spin single": 800,
        "t-spin mini double": 400,
        "t-spin double": 1200,
        "t-spin triple": 1600, 
        "no lines": 0
    }

    score_list = ["single", "double", "triple", "tetris", "t-spin mini no lines", "t-spin no lines", 
        "t-spin mini single", "t-spin single", "t-spin mini double", "t-spin double", "t-spin triple"]

    difficult_list = ["tetris", "t-spin mini single", "t-spin single", "t-spin mini double", "t-spin double", "t-spin triple"]


scorefile = os.path.join(os.path.dirname(__file__), ".highscores")

def load_score():
    """ Returns the highest score, or 0 if no one has scored yet """
    try:
        with open(scorefile) as file:
            scores = sorted([int(score.strip())
                             for score in file.readlines()
                             if score.strip().isdigit()], reverse=True)
    except IOError:
        scores = []

    return scores[0] if scores else 0

def write_score(score):
    """
    Writes score to file.
    """
    assert str(score).isdigit()
    with open(scorefile, 'a') as file:
        file.write("{}\n".format(score))