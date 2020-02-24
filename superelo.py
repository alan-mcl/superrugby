import argparse
from functools import total_ordering

import numpy as np
from skopt import gp_minimize

import srdb


#--------------------------------------------
def bruRound(margin, default_win_margin=6):
	return max(6, int(round(margin/3.0)*3.0)) * np.sign(margin)

#--------------------------------------------
def bruRound2(margin, default_win_margin=6):
	
	if abs(margin) < 6:
		return default_win_margin * np.sign(margin)
	else:
		return round(margin)

#--------------------------------------------
# traditional ELO functions
def expected(A, B):
	"""
	Calculate expected score of A in a match against B

	:param A: Elo rating for player A
	:param B: Elo rating for player B
	"""
	return 1 / (1 + 10 ** ((B - A) / 400))

#--------------------------------------------
def elo(old, exp, score, k=32):
	"""
	Calculate the new Elo rating for a player

	:param old: The previous Elo rating
	:param exp: The expected score for this match
	:param score: The actual score for this match
	:param k: The k-factor for Elo (default: 32)
	"""
	return old + k * (score - exp)

#--------------------------------------------
@total_ordering
class Date:
	"""
	Creates a cyclic date object with year and week
	attributes.

	date.next and date.prev function calls increment
	and decrement the date object.

	"""
	def __init__(this, year, week):
		this.nweeks = 17
		this.year = year
		this.week = week

	def __eq__(this, other):
		return (this.year, this.week) == (other.year, other.week)

	def __lt__(this, other):
		return (this.year, this.week) < (other.year, other.week)

	def __sub__(this, other):
		dy = (this.year - other.year)
		dw = (this.week - other.week)
		return dy*this.nweeks + dw

	@property
	def next(this):
		if this.week < this.nweeks:
			return Date(this.year, this.week + 1)
		else:
			return Date(this.year + 1, 1)

	@property
	def prev(this):
		if this.week > 1:
			return Date(this.year, this.week - 1)
		else:
			return Date(this.year - 1, this.nweeks)
			
#--------------------------------------------
def getActualDiff(home_points, away_points):
	home_perc = home_points / (home_points + away_points*1.0)
	away_perc = away_points / (away_points + home_points*1.0)
	
	return home_perc - away_perc
	
#--------------------------------------------
def polyfit(elo_values, actual_margins, polyorder=3):
	
	return np.polyfit(elo_values, actual_margins, polyorder)

#==============================================================================
class Model:

	def getName(this):
		pass
		
	def getStartingElo(this):
		pass
	
	def getPredictedMargin(this, game, home_elo, away_elo):
		pass
		
	def getNewElo(this, old_elo, expected, points_for, points_against):
		pass
		
	def newSeason(this, elo, yr_elo_diffs, yr_actual_margins, elo_diffs, actual_margins):
		pass
		
#==============================================================================
class LinearLSModel(Model):

	def __init__(this, kfactor=32, default_win_margin=6):
		this.kfactor = kfactor
		this.m0 = 4.490665
		this.m1 = 49.64006
		this.default_win_margin = default_win_margin
		
	def getName(this):
		return "Linear Least Squares"
	
	def getStartingElo(this):
		return 1500
		
	def getPredictedMargin(this, game, home_elo, away_elo):
		
		# traditional ELO calc
		home_perc = expected(home_elo, away_elo)
		away_perc = expected(away_elo, home_elo)
		
		d = home_perc - away_perc
		
		pred_margin = this.m0 + this.m1*d
		
		return (pred_margin, home_perc, away_perc)
				
	def getNewElo(this, old_elo, expected, points_for, points_against):
		actual = points_for / (points_for + points_against*1.0)
				
		return elo(old_elo, expected, actual, this.kfactor)
	
	def newSeason(this, elo, yr_elo_diffs, yr_actual_margins, elo_diffs, actual_margins):
		pass
		
#==============================================================================
class RoundedLLSModel(LinearLSModel):

	def getName(this):
		return "Linear LS (BRUround2, seasonal mean reg, k=%i, dwm=%i)" % (this.kfactor, this.default_win_margin)
	
	def getPredictedMargin(this, game, home_elo, away_elo):
		
		(pred_margin, home_perc, away_perc) = LinearLSModel.getPredictedMargin(this, game, home_elo, away_elo)
		
		return (bruRound2(pred_margin, this.default_win_margin), home_perc, away_perc)
		
	def newSeason(this, elo, yr_elo_diffs, yr_actual_margins, elo_diffs, actual_margins):
		for team in elo.keys():
			rtg = elo[team]
			elo[team] = round(1500 + 0.9 * (rtg - 1500))
		
#==============================================================================
class RoundedLLSModel2(LinearLSModel):

	def getName(this):
		return "Linear LS (BRUround2, seasonal mean reg, k=%i, dwm=%i, annual refit)" % (this.kfactor, this.default_win_margin)
	
	def getPredictedMargin(this, game, home_elo, away_elo):
		
		(pred_margin, home_perc, away_perc) = LinearLSModel.getPredictedMargin(this, game, home_elo, away_elo)
		
		return (bruRound2(pred_margin, this.default_win_margin), home_perc, away_perc)
		
	def newSeason(this, elo, yr_elo_diffs, yr_actual_margins, elo_diffs, actual_margins):
		for team in elo.keys():
			rtg = elo[team]
			elo[team] = round(1500 + 0.9 * (rtg - 1500))
			
		if len(yr_elo_diffs) > 0:
			(m1, m0) = polyfit(elo_diffs, actual_margins, 1)
			this.m1 = m1
			this.m0 = m0		
			
#==============================================================================
class QuadraticModel(Model):

	def __init__(this, kfactor=32, regress=0.8, default_win_margin=6):
		this.kfactor = kfactor
		this.m0 = 3.7094
		this.m1 = 39.394
		this.m2 = 10.114
		this.m3 = 58.82
		this.d_min = 0
		this.default_win_margin = default_win_margin
		
	def getName(this):
		return "3rd Order Poly"
	
	def getStartingElo(this):
		return 1500
		
	def getPredictedMargin(this, game, home_elo, away_elo):
		
		# traditional ELO calc
		home_perc = expected(home_elo, away_elo)
		away_perc = expected(away_elo, home_elo)
		
		d = home_perc - away_perc
		
		#if abs(d) < this.d_min:
		#	pred_margin = bruRound2(-exp(-6.086877441*d)+1)
		#else:
		pred_margin = this.m0 + this.m1*d + this.m2*d*d + this.m3*d*d*d
		
		return (pred_margin, home_perc, away_perc)
				
	def getNewElo(this, old_elo, expected, points_for, points_against):
		actual = points_for / (points_for + points_against*1.0)
				
		return elo(old_elo, expected, actual, this.kfactor)
	
	def newSeason(this, elo, yr_elo_diffs, yr_actual_margins, elo_diffs, actual_margins):
		pass


#==============================================================================
class RoundedQuadraticModel(QuadraticModel):

	def getName(this):
		return "3rd Order Poly (BRU rounding)"
	
	def getPredictedMargin(this, game, home_elo, away_elo):
		
		(pred_margin, home_perc, away_perc) = QuadraticModel.getPredictedMargin(this, game, home_elo, away_elo)
		
		if abs(home_perc - away_perc) < this.d_min:
			return (round(pred_margin), home_perc, away_perc)
		else:
			return (bruRound(pred_margin), home_perc, away_perc)

#==============================================================================
class RoundedQuadraticModel2(QuadraticModel):

	def __init__(this, d_min=0.0, kfactor=32, default_win_margin=6):
		QuadraticModel.__init__(this)
		this.d_min = d_min
		this.kfactor = kfactor
		this.default_win_margin = default_win_margin

	def getName(this):
		return "3rd Order Poly (BRUround2, k=%i, dmin=%.2f)" % (this.kfactor, this.d_min)
	
	def getPredictedMargin(this, game, home_elo, away_elo):
		
		(pred_margin, home_perc, away_perc) = QuadraticModel.getPredictedMargin(this, game, home_elo, away_elo)
		
		if abs(home_perc - away_perc) < this.d_min:
			return (round(pred_margin), home_perc, away_perc)
		else:
			return (bruRound2(pred_margin, this.default_win_margin), home_perc, away_perc)


#==============================================================================
class RoundedQuadraticModel3(RoundedQuadraticModel2):

	def __init__(this, d_min=0.0, kfactor=32, default_win_margin=6):
		QuadraticModel.__init__(this)
		this.d_min = d_min
		this.kfactor = kfactor
		this.default_win_margin = default_win_margin

	def getName(this):
		return "3rd Order Poly (BRUround2, seasonal mean reg, k=%i, dmin=%.2f, dwm=%i)" % (this.kfactor, this.d_min, this.default_win_margin)
	
	def newSeason(this, elo, yr_elo_diffs, yr_actual_margins, elo_diffs, actual_margins):
		for team in elo.keys():
			rtg = elo[team]
			elo[team] = round(1500 + 0.9 * (rtg - 1500))
			
#==============================================================================
class RoundedQuadraticModel4(RoundedQuadraticModel3):

	def __init__(this, d_min=0.0, kfactor=32, default_win_margin=6):
		QuadraticModel.__init__(this)
		this.d_min = d_min
		this.kfactor = kfactor
		this.default_win_margin = default_win_margin

	def getName(this):
		return "3rd Order Poly (BRUround2, seasonal mean reg, k=%i, dmin=%.2f, dwm=%i, seasonal refit)" % (this.kfactor, this.d_min, this.default_win_margin)
	
	def getPredictedMargin(this, game, home_elo, away_elo):
		
		(pred_margin, home_perc, away_perc) = QuadraticModel.getPredictedMargin(this, game, home_elo, away_elo)
		
		if abs(home_perc - away_perc) < this.d_min:
			tm = TiersModel()
			return tm.getPredictedMargin(game, home_elo, away_elo)
		else:
			return (bruRound2(pred_margin, this.default_win_margin), home_perc, away_perc)

	def newSeason(this, elo, yr_elo_diffs, yr_actual_margins, elo_diffs, actual_margins):
		RoundedQuadraticModel3.newSeason(this, elo, yr_elo_diffs, yr_actual_margins, elo_diffs, actual_margins)

		if len(yr_elo_diffs) > 0:
			(m3, m2, m1, m0) = polyfit(elo_diffs, actual_margins, 3)
			this.m3 = m3
			this.m2 = m2
			this.m1 = m1
			this.m0 = m0

#==============================================================================
class TiersModel(Model):

	def __init__(this):
		this.tiers = \
		{
			"Crusaders" : 1,
			
			"Lions" : 2,
			"Hurricanes" : 2,
			"Highlanders" : 2,
			"Chiefs" : 2,

			"Waratahs" : 3,
			"Sharks" : 3,
			"Stormers" : 3,
			"Bulls" : 3,
			"Blues" : 3,
			"Brumbies" : 3,
			"Reds" : 3,

			"Jaguares" : 3.5,

			"Rebels" : 4,
			"Force" : 4,
			"Cheetahs" : 4,

			"Sunwolves" : 5,
			"Kings" : 5,
		}
		
	def getName(this):
		return "TiersModel"

	def getStartingElo(this):
		return 1500
	
	def getPredictedMargin(this, game, home_elo, away_elo):
		
		ht = this.tiers[game.home_team]
		at = this.tiers[game.away_team]
		
		base = 6
		
		margin = base
		
		if ht < at:
			diff = at - ht
			margin = base + base*pow(diff-1, 2)
			
		elif ht > at:
			diff = ht - at
			margin = -(base + base*pow(diff-1, 2))
			
		else:
			margin = base
			
		return (margin, 0.0, 0.0)
			
		
	def getNewElo(this, old_elo, expected, points_for, points_against):
		return 1500
		
	def newSeason(this, elo, yr_elo_diffs, yr_actual_margins, elo_diffs, actual_margins):
		pass

#==============================================================================
class OddsModel(Model):

	def __init__(this):
		pass

	def getName(this):
		return "OddsModel"

	def getStartingElo(this):
		return 1500

	def getPredictedMargin(this, game, home_elo, away_elo):

		hlc = game.home_line_close

		# no odds: return home team by 3
		if hlc is None or hlc == 0:
			return (3, 0.5, 0.5)

		# hack this to get kinda sensible results
		if hlc < 0:
			away_perc = 1.0/abs(hlc)
			home_perc = 1.0-away_perc
		else:
			home_perc = 1.0/abs(hlc)
			away_perc = 1.0-home_perc

		return (bruRound2(-hlc, 3), home_perc, away_perc)

	def getNewElo(this, old_elo, expected, points_for, points_against):
		return 1500

	def newSeason(this, elo, yr_elo_diffs, yr_actual_margins, elo_diffs, actual_margins):
		pass
		
#==============================================================================		
def closeGameAnalysis(close_games):
	
	residuals = []
	hlc_residuals = []
	wins = 0
	mp = 0
	bonus = 0
	bets = 0
	betwins = 0
	betwinamt = 0.0

	for game_tuple in close_games:
		date = game_tuple[0]
		pred_margin = game_tuple[1]
		bet = game_tuple[2]
		betwin = game_tuple[3]
		betamt = game_tuple[4]
		game = game_tuple[5]

		bets += bet
		betwins += betwin

		if betwin == 1:
			betwinamt += betamt * 1.94

		actual_margin = game.home_score - game.away_score
		error = actual_margin - pred_margin
		residuals.append(error)

		if game.home_line_close is not None:
			hlc_error = (-game.home_line_close) - pred_margin
			hlc_residuals.append(hlc_error)
		
		if np.sign(pred_margin) == np.sign(actual_margin) and actual_margin != 0:
			wins += 1
		
		if error <= 5 and error >= -5:
			mp += 1
			
		if round(error) == 0:
			bonus += 1
	
	summary("CLOSE", residuals, wins, mp, bonus, hlc_residuals, bets, betwins, betwinamt)

#==============================================================================
def runModel(model, verbose=False, data_output=False, bet_output=False):
	elo = {}
	residuals = []
	close_games = []
	wins = 0.0
	margin_pts = 0
	bonus_pts = 0
	elo_diffs = []
	actual_margins = []
	hlc_redisuals = []
	bets = 0
	betwins = 0
	betwinamt = 0.0

	# 1995: SUPER12
	# 2009: first year of data in srdb
	# 2016: Jap/Arg expansion
	FIRST_YEAR_TO_REPORT = 2016
	FIRST_YEAR_TO_REPORT = 2010
	CLOSE_GAME_D = 0.05
	
	yr = FIRST_YEAR_TO_REPORT
	yr_residuals = []
	yr_wins = 0
	yr_margin_pts = 0
	yr_bonus_pts = 0
	yr_elo_diffs = []
	yr_actual_margins = []
	yr_hlc_residuals = []
	yr_bets = 0
	yr_betwins = 0
	yr_betwinamt = 0.0

	print "~"*20
	print model.getName()
	print "~"*20

	# loop over historical games in chronological order
	year = 0
	week = 0
	for game in sorted(srdb.games, key=lambda g: Date(g.year, g.week)):
	
		if game.year > year:
			model.newSeason(elo, yr_elo_diffs, yr_actual_margins, elo_diffs, actual_margins)
			year = game.year
			week = 0
			if verbose:
				print "==================SEASON: %i" % (year)
			
		if game.week > week:
			week = game.week
			if verbose:
				print "--------week: %i" % (week)
			
		home_elo = away_elo = model.getStartingElo();
		
		if elo.has_key(game.home_team):
			home_elo = elo[game.home_team]
		if elo.has_key(game.away_team):
			away_elo = elo[game.away_team]
		
		(pred_margin, home_perc, away_perc) = model.getPredictedMargin(game, home_elo, away_elo)
		if game.finished:
			actual_margin = game.home_score-game.away_score
		else:
			actual_margin = 0
			
		d = home_perc-away_perc

		bet = 0
		betwin = 0
		betamt = 0
		beton = shouldBet(game, pred_margin)
		if beton is not None:
			bet = 1
		if game.home_line_close is not None and beton is not None:

			distance = abs(abs(pred_margin)-abs(game.home_line_close))

			if distance > 20:
				betamt = 3
			elif distance > 10:
				betamt = 2
			else:
				betamt = 1

			betwinamt -= betamt
			yr_betwinamt -= betamt

			if actual_margin > 0:
				if actual_margin > -game.home_line_close:
					# predict home team by 3, bookies handicap them by 2. bet on home
					if beton == game.home_team:
						betwin = 1
				else:
					# predict home team by 3, bookies handicap them by 4. bet on away
					if beton == game.away_team:
						betwin = 1
			elif pred_margin < 0:
				if -pred_margin > game.home_line_close:
					# predict away team by 3, bookies handicap them by 2. bet on away
					if beton == game.away_team:
						betwin = 1
				else:
					# predict away team by 3, bookies handicap them by 4. bet on home
					if beton == game.home_team:
						betwin = 1

			if betwin == 1:
				betwinamt += betamt * 1.94
				yr_betwinamt += betamt * 1.94

		bets += bet
		betwins += betwin
		yr_bets += bet
		yr_betwins += betwin

		# ----------- printout stuffs
		hlc = "?"
		betstr = "-"
		if game.home_line_close is not None:
			hlc = str(game.home_line_close)
		if bet == 1:
			if betwin == 1:
				betstr = "W"
			else:
				betstr = "L"
		if verbose:
			print "week %i: %s: %s(%i) v %s(%i): p %d a %i hlc %s beton %s(%d,%s)" % \
				(game.week, game.date, game.home_team, home_elo, game.away_team, away_elo, pred_margin, actual_margin, hlc, beton, betamt, betstr)
		elif not game.finished:
			print "week %i: %s: %s(%i) v %s(%i): predicted %d hlc %s beton %s(%d,%s)" % (game.week, game.date, game.home_team, home_elo, game.away_team, away_elo, pred_margin, hlc, beton, betamt, betstr)
		elif data_output and year > 2009:
			print "%i,%i,%s,%s,%f,%i" % (year, game.week, game.home_team, game.away_team, d, actual_margin)
		elif bet_output:
			if bet == 1:
				print "week %i: %s: %s(%i) v %s(%i): p %d a %i hlc %s beton %s(%d,%s)" % \
					(game.week, game.date, game.home_team, home_elo, game.away_team, away_elo, pred_margin, actual_margin, hlc, beton, betamt, betstr)
		# ----------- printout stuffs
				
		if game.finished:
			new_home_elo = model.getNewElo(home_elo, home_perc, game.home_score, game.away_score)
			new_away_elo = model.getNewElo(away_elo, away_perc, game.away_score, game.home_score)

			elo[game.home_team] = new_home_elo
			elo[game.away_team] = new_away_elo
			
			if abs(d) < CLOSE_GAME_D:
				close_games.append( (d, pred_margin, bet, betwin, betamt, game) )
			
		# one year burn in
		if year > FIRST_YEAR_TO_REPORT-1:

			if year > yr:
				if verbose or not data_output:
					summary(str(yr), yr_residuals, yr_wins, yr_margin_pts, yr_bonus_pts, yr_hlc_residuals, yr_bets, yr_betwins, yr_betwinamt)
					if verbose:
						print "processing %s..." % (str(year))
				yr = year
				yr_residuals = []
				yr_wins = 0
				yr_margin_pts = 0
				yr_bonus_pts = 0
				yr_elo_diffs = []
				yr_actual_margins = []
				yr_hlc_residuals = []
				yr_bets = 0
				yr_betwins = 0
				yr_betwinamt = 0

			if game.finished:
				error = actual_margin - pred_margin
				residuals.append(error)
				yr_residuals.append(error)
				if game.home_line_close is not None:
					hlc_error = (-game.home_line_close) - pred_margin
					hlc_redisuals.append(hlc_error)
					yr_hlc_residuals.append(hlc_error)

				elo_diffs.append(d)
				actual_margins.append(d)

				yr_elo_diffs.append(d)
				yr_actual_margins.append(actual_margin)

				if np.sign(pred_margin) == np.sign(actual_margin) and actual_margin != 0:
					wins += 1
					yr_wins += 1

				if error <= 5 and error >= -5:
					margin_pts += 1
					yr_margin_pts += 1

				if round(error) == 0:
					bonus_pts += 1
					yr_bonus_pts += 1
				
	if verbose or not data_output:
		summary(str(yr), yr_residuals, yr_wins, yr_margin_pts, yr_bonus_pts, yr_hlc_residuals, yr_bets, yr_betwins, yr_betwinamt)
		summary("ALL ", residuals, wins, margin_pts, bonus_pts, hlc_redisuals, bets, betwins, betwinamt)
		
	closeGameAnalysis(close_games)
		
	return residuals

def shouldBet(game, pred_margin):
	if game.home_line_close is not None:

		if abs(pred_margin) > 1 and \
				4 < abs(abs(pred_margin)-abs(game.home_line_close)) < 150:
			if pred_margin > 0:
				if pred_margin > -game.home_line_close:
					# predict home team by 3, bookies handicap them by 2. bet on home
					return game.home_team
				else:
					# predict home team by 3, bookies handicap them by 4. bet on away
					return game.away_team
			elif pred_margin < 0:
				if -pred_margin > game.home_line_close:
					# predict away team by 3, bookies handicap them by 2. bet on away
					return game.away_team
				else:
					# predict away team by 3, bookies handicap them by 4. bet on home
					return game.home_team

	return None

def summary(name, residuals, wins, margin_pts, bonus_pts, hlc_residuals, bets, betwins, betwinamt):

	if len(residuals) == 0:
		return

	# mean = np.mean(residuals)
	# stddev = np.std(residuals)
	mdm = np.abs(residuals).mean()
	mse = np.square(residuals).mean()
	winp = 100.0*wins/len(residuals)
	bru = wins+margin_pts*0.5
	# hle_mse = np.square(hlc_residuals).mean()

	betp = 100.0*bets/len(residuals)
	betwinp = 0.0
	if bets > 0:
		betwinp = 100.0*betwins/bets

	print "%s (n=%i): mdm[%4.1f] mse[%4.1f] win[%3.1f%%], m%%[%3.1f%%] b%%[%3.1f%%] BRU[%.1f] bets[%d] betwin%%[%.1f%%] betwinamt[%.1f]" % \
		  (name, len(residuals), mdm, mse, winp, 100.0*margin_pts/len(residuals), 100.0*bonus_pts/len(residuals), bru, bets, betwinp, betwinamt)

def optimize(model_param):
	"""
	Function to optimize model hyper-parameters

	"""
	def obj(parameters):
		"""
		Evaluates the mean absolute error for a set of input
		parameters: kfactor, decay, regress.

		"""
		kfactor, regress = parameters
		
		model = None
		
		if model_param == "3op":
			model = RoundedQuadraticModel(kfactor, regress)
		else:
			print "TODO OPTIMIZE THIS MODEL NOT SUPPORTED"
			exit
			
		residuals = runModel(model, False)
		
		mean_abs_error = np.abs(residuals).mean()
		return mean_abs_error

	bounds = [(25., 85.), (0.4, 0.8)]
	res_gp = gp_minimize(obj, bounds, n_calls=100, verbose=True)

	print("Best score: {:.4f}".format(res_gp.fun))
	print("Best parameters: {}".format(res_gp.x))


#----------------------------
def main():

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"model",
		action="store",
		type=str,
		help="'all' for all, 'lls' for Linear LS, '3op' for Quadratic, 'best' for current best")
	parser.add_argument(
		"--v",
		action="store_true",
		default=False,
		help="verbose mode")
	parser.add_argument(
		"--o",
		action="store_true",
		default=False,
		help="optimize")
	parser.add_argument(
		"--d",
		action="store_true",
		default=False,
		help="output data CSV")
	parser.add_argument(
		"--b",
		action="store_true",
		default=False,
		help="output bets")

	args = parser.parse_args()
	args_dict = vars(args)
	
	verbose = args_dict['v']
	data_output = args_dict['d']
	bet_output = args_dict['b']
	
	m = args_dict['model']
	
	if args_dict['o']:
		optimize(m)
	else:
		if m == "lls":
			runModel(LinearLSModel(), verbose, data_output, bet_output)
			runModel(RoundedLLSModel(), verbose, data_output, bet_output)
			runModel(RoundedLLSModel2(kfactor=50, default_win_margin=3), verbose, data_output, bet_output)
		elif m == "3op":
			runModel(RoundedQuadraticModel3(kfactor=50, default_win_margin=3), verbose, data_output, bet_output)
			runModel(RoundedQuadraticModel4(kfactor=50, default_win_margin=3), verbose, data_output, bet_output)
		elif m == "all":
			runModel(RoundedQuadraticModel4(kfactor=50, default_win_margin=3, d_min=0.1), verbose, data_output, bet_output)
			runModel(RoundedQuadraticModel3(kfactor=50, default_win_margin=3), verbose, data_output, bet_output)
			runModel(RoundedLLSModel(kfactor=50, default_win_margin=3), verbose, data_output, bet_output)
			runModel(TiersModel(), verbose, data_output, bet_output)
		elif m == "odds":
			runModel(RoundedQuadraticModel3(kfactor=50, default_win_margin=3), verbose, data_output, bet_output)
			runModel(OddsModel(), verbose, data_output, bet_output)
		elif m == "best":
			runModel(RoundedQuadraticModel3(kfactor=50, default_win_margin=3), verbose, data_output, bet_output)
			

if __name__ == "__main__":
	main()
