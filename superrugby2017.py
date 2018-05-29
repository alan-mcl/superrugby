import elo

# starting elo (https://www.reddit.com/r/rugbyunion/comments/4s4sur/elo_ratings_for_super_rugby_teams/)
# TEAM = (ELO, AVG POINTS PER GAME)
teams = \
	{
	"Highlanders" : (1598,28.1),
	"Lions" : (1593,35.6),
	"Chiefs" : (1586,32.7),
	"Crusaders" : (1596,32.4),
	"Hurricanes" : (1571,30.53),
	"Sharks" : (1550,24.0),
	"Brumbies" : (1540,28.3),
	"Blues" : (1516,24.9),
	"Waratahs" : (1544,27.5),
	"Stormers" : (1542,29.3),
	"Bulls" : (1472,26.6),
	"Rebels" : (1457,24.3),
	"Reds" : (1418,19.3),
	"Jaguares" : (1461,25.0),
	"Force" : (1418,17.3),
	"Cheetahs" : (1424,25.13),
	"Kings" : (1359,18.8),
	"Sunwolves" : (1345,19.5),
	}

matches = \
	[
	("Rebels","Blues"),
	("Highlanders","Chiefs"),
	("Reds","Sharks"),
	("Sunwolves","Hurricanes"),
	("Crusaders","Brumbies"),
	("Waratahs","Force"),
	("Cheetahs","Lions"),
	("Kings","Jaguares"),
	("Stormers","Bulls"),
	]

for match in matches:
	
	# home team
	team1 = match[0]
	
	# away team
	team2 = match[1]

	team1_expected = elo.expected(teams[team1][0], teams[team2][0])
	team2_expected = elo.expected(teams[team2][0], teams[team1][0])
	
	# assume avg points scored
	
	team1_tot_pts = teams[team1][1]
	team2_tot_pts = teams[team2][1]
	
	team1_predicted_score = team1_tot_pts * team1_expected
	team2_predicted_score = team2_tot_pts * team2_expected
	diff = team1_predicted_score - team2_predicted_score
	
	if diff > 0:
		winner = team1
	else:
		winner = team2
	
	print "%s (%s) vs %s (%s): %s by %s" % (team1, team1_expected, team2, team2_expected, winner, abs(diff))
	