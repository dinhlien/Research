# try:
#     from pip import main as pipmain
# except:
#     from pip._internal import main as pipmain
#     pipmain(['install','gekko'])

from gekko import GEKKO

# Initialize Model
m = GEKKO(remote=True)
