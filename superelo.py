import argparse
from collections import defaultdict
from functools import total_ordering
from math import sqrt
from math import exp

import numpy as np
from scipy.optimize import minimize
from scipy.special import erf, erfinv
from skopt import gp_minimize

import srdb

def bruRound(margin, default_win_margin=6):
	return max(6, int(round(margin/3.0)*3.0)) * np.sign(margin)

def bruRound2(margin, default_win_margin=6):
	
	if abs(margin) < 6:
		return default_win_margin * np.sign(margin)
	else:
		return round(margin)
		
	#return max(6, round(margin)) * np.sign(margin)

# traditional ELO functions
def expected(A, B):
	"""
	Calculate expected score of A in a match against B

	:param A: Elo rating for player A
	:param B: Elo rating for player B
	"""
	return 1 / (1 + 10 ** ((B - A) / 400))

def elo(old, exp, score, k=32):
	"""
	Calculate the new Elo rating for a player

	:param old: The previous Elo rating
	:param exp: The expected score for this match
	:param score: The actual score for this match
	:param k: The k-factor for Elo (default: 32)
	"""
	return old + k * (score - exp)
	
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

#----------------------------
class Model:

	def getStartingElo(this):
		pass
	
	def getPredictedMargin(this, home_elo, away_elo):
		pass
		
	def getNewElo(this, old_elo, expected, points_for, points_against):
		pass
		
	def newSeason(this, elo):
		pass
		
#----------------------------
class MeloModel(Model):

	def __init__(this, kfactor=32, regress=0.8):
		
		this.hfa = 54 # home field advantage
		this.sigma = 300
		this.smooth = 5.
		#this.kfactor = 85
		this.kfactor = kfactor
		
		# handicap range
		this.hcap_range = np.arange(-50.5, 51.5, 1)
		
	def getName(this):
		return "MElo"

	def getStartingElo(this):
		result = {}
		for hcap in this.hcap_range:
			result[hcap] = 1500
		return result
		
	def norm_cdf(this, x, loc=0, scale=1):
		"""
		Normal cumulative probability distribution

		"""
		return 0.5*(1 + erf((x - loc)/(sqrt(2)*scale)))
		
	def win_prob(this, rtg_diff):
		
		"""
		Win probability as function of ELO difference

		"""
		return this.norm_cdf(rtg_diff, scale=this.sigma)
		
	def cdf(this, home_elo, away_elo):
		"""
		Cumulative (integrated) probability that,
		score home - score away > x.

		"""
		cprob = []
		
		for hcap in this.hcap_range:
			home_rtg = home_elo[hcap]
			away_rtg = away_elo[hcap]

			elo_diff = home_rtg - away_rtg + this.hfa

			win_prob = this.win_prob(elo_diff)
			cprob.append(win_prob)

		return this.hcap_range, sorted(cprob, reverse=True)
		
	def elo_change(this, rating_diff, points, hcap):
		"""
		Change in home team ELO rating after a single game

		"""
		sign = np.sign(hcap)
		
		TINY = 1e-3

		prior = this.win_prob(rating_diff)
		
		exp_arg = (points - hcap)/max(this.smooth, TINY)
		
		if this.smooth:
			post = this.norm_cdf(points, loc=hcap, scale=this.smooth)
		else:
			post = (1 if points > hcap else 0)
		
		return this.kfactor * (post - prior)
	
	def getPredictedMargin(this, home_elo, away_elo):
		"""
		Predict the spread/total for a matchup, given current
		knowledge of each team's Elo ratings.
		"""
		# cumulative point distribution
		points, cprob = this.cdf(home_elo, away_elo)

		home_bounty = {}
		away_bounty = {}
		
		for hcap in this.hcap_range:
			
			# query current elo ratings from most recent game
			home_rtg = home_elo[hcap]
			away_rtg = away_elo[hcap]

			# elo change when home(away) team is handicapped
			rtg_diff = home_rtg - away_rtg + this.hfa
			bounty = this.elo_change(rtg_diff, 0, hcap)
			
			home_bounty[hcap] = bounty
			away_bounty[hcap] = -bounty

		# plot median prediction (compare to vegas spread/total)
		index = np.square([p - 0.5 for p in cprob]).argmin()
		if index in range(1, len(cprob) - 2):
			x0, y0 = (points[index - 1], cprob[index - 1])
			x1, y1 = (points[index], cprob[index])
			x2, y2 = (points[index + 1], cprob[index + 1])

			# fit a quadratic polynomial
			coeff = np.polyfit([x0, x1, x2], [y0, y1, y2], 2)
			res = minimize(
					lambda x: np.square(np.polyval(coeff, x) - 0.5), x1
				  )

			return (0.5 * round(res.x * 2), home_bounty, away_bounty)
			
		return (points[index], home_bounty, away_bounty)

		
	def getNewElo(this, old_elo, bounty, points_for, points_against):
		result = {}
		for hcap in this.hcap_range:
			result[hcap] = old_elo[hcap] + bounty[hcap]
			
		return result
		
	def newSeason(this, elo):
		pass
		
#----------------------------
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
		
	def getPredictedMargin(this, home_elo, away_elo):
		
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
	
	def newSeason(this, elo):
		pass

#----------------------------
class QuadraticModel2(QuadraticModel):
	def __init__(this, kfactor=32, regress=0.8):
		this.kfactor = kfactor
		this.m0 = 4.237
		this.m1 = 48.125
		this.m2 = 8.4393
		this.m3 = 93.726

	def getName(this):
		return "3rd Order Poly (new params)"
		
#----------------------------
class RoundedQuadraticModel(QuadraticModel):

	def getName(this):
		return "3rd Order Poly (BRU rounding)"
	
	def getPredictedMargin(this, home_elo, away_elo):
		
		(pred_margin, home_perc, away_perc) = QuadraticModel.getPredictedMargin(this, home_elo, away_elo)
		
		if abs(home_perc - away_perc) < this.d_min:
			return (round(pred_margin), home_perc, away_perc)
		else:
			return (bruRound(pred_margin), home_perc, away_perc)

#----------------------------
class RoundedQuadraticModel2(QuadraticModel):

	def __init__(this, d_min=0.0, kfactor=32, default_win_margin=6):
		QuadraticModel.__init__(this)
		this.d_min = d_min
		this.kfactor = kfactor
		this.default_win_margin = default_win_margin

	def getName(this):
		return "3rd Order Poly (BRUround2, k=%i, dmin=%.2f)" % (this.kfactor, this.d_min)
	
	def getPredictedMargin(this, home_elo, away_elo):
		
		(pred_margin, home_perc, away_perc) = QuadraticModel.getPredictedMargin(this, home_elo, away_elo)
		
		if abs(home_perc - away_perc) < this.d_min:
			return (round(pred_margin), home_perc, away_perc)
		else:
			return (bruRound2(pred_margin, this.default_win_margin), home_perc, away_perc)


#----------------------------
class RoundedQuadraticModel3(RoundedQuadraticModel2):

	def __init__(this, d_min=0.0, kfactor=32, default_win_margin=6):
		QuadraticModel.__init__(this)
		this.d_min = d_min
		this.kfactor = kfactor
		this.default_win_margin = default_win_margin

	def getName(this):
		return "3rd Order Poly (BRUround2, seasonal mean reg, k=%i, dmin=%.2f, dwm=%i)" % (this.kfactor, this.d_min, this.default_win_margin)
	
	def newSeason(this, elo):
		for team in elo.keys():
			rtg = elo[team]
			elo[team] = round(1500 + 0.9 * (rtg - 1500))

#----------------------------
class LinearLSModel(Model):

	def __init__(this):
		this.kfactor = 32
		this.m0 = 4.490665
		this.m1 = 49.64006
		
	def getName(this):
		return "Linear Least Squares"
	
	def getStartingElo(this):
		return 1500
		
	def getPredictedMargin(this, home_elo, away_elo):
		
		# traditional ELO calc
		home_perc = expected(home_elo, away_elo)
		away_perc = expected(away_elo, home_elo)
		
		d = home_perc - away_perc
		
		pred_margin = this.m0 + this.m1*d
		
		return (pred_margin, home_perc, away_perc)
				
	def getNewElo(this, old_elo, expected, points_for, points_against):
		actual = points_for / (points_for + points_against*1.0)
				
		return elo(old_elo, expected, actual, this.kfactor)
	
	def newSeason(this, elo):
		pass
		
#----------------------------
class RoundedLLSModel(LinearLSModel):

	def getName(this):
		return "Linear LS (BRU rounding)"
	
	def getPredictedMargin(this, home_elo, away_elo):
		
		(pred_margin, home_perc, away_perc) = LinearLSModel.getPredictedMargin(this, home_elo, away_elo)
		
		return (bruRound(pred_margin), home_perc, away_perc)
	
#----------------------------
def getActualDiff(home_points, away_points):
	home_perc = home_points / (home_points + away_points*1.0)
	away_perc = away_points / (away_points + home_points*1.0)
	
	return home_perc - away_perc

#----------------------------
def runModel(model, verbose=False, data_output=False):
	elo = {}
	residuals = []
	wins = 0.0
	margin_pts = 0
	bonus_pts = 0
	
	# 1995: SUPER12
	# 2009: first year of data in srdb
	# 2016: Jap/Arg expansion
	FIRST_YEAR_TO_REPORT = 2016
	yr = FIRST_YEAR_TO_REPORT
	yr_residuals = []
	yr_wins = 0
	yr_margin_pts = 0
	yr_bonus_pts = 0
	
	print "~"*20
	print model.getName()
	print "~"*20

	# loop over historical games in chronological order
	year = 0
	week = 0
	for game in sorted(srdb.games, key=lambda g: Date(g.year, g.week)):
	
		if game.year > year:
			model.newSeason(elo)
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
		
		(pred_margin, home_perc, away_perc) = model.getPredictedMargin(home_elo, away_elo)
		if game.finished:
			actual_margin = game.home_score-game.away_score
		else:
			actual_margin = 0
			
		d = home_perc-away_perc
		
		if verbose or not game.finished:
			if model.getName() == "MElo":
				print "%s(%i) v %s(%i): predicted %d actual %i" % (game.home_team, home_elo[.5], game.away_team, away_elo[.5], pred_margin, actual_margin)
			else:
				print "%s(%i) v %s(%i): predicted %d actual %i" % (game.home_team, home_elo, game.away_team, away_elo, pred_margin, actual_margin)
		elif data_output and year > 2009:
			print "%f,%f,%i" % (d, getActualDiff(game.home_score, game.away_score), actual_margin)
				
		if game.finished:
			new_home_elo = model.getNewElo(home_elo, home_perc, game.home_score, game.away_score)
			new_away_elo = model.getNewElo(away_elo, away_perc, game.away_score, game.home_score)

			elo[game.home_team] = new_home_elo
			elo[game.away_team] = new_away_elo

			# one year burn in
			if year > FIRST_YEAR_TO_REPORT-1:
				
				if year > yr:
					if verbose or True:
						summary(str(yr), yr_residuals, yr_wins, yr_margin_pts, yr_bonus_pts)
						yr = year
						yr_residuals = []
						yr_wins = 0
						yr_margin_pts = 0
						yr_bonus_pts = 0
								
				error = actual_margin - pred_margin
				residuals.append(error)
				yr_residuals.append(error)

				if np.sign(pred_margin) == np.sign(actual_margin) and actual_margin != 0:
					wins += 1
					yr_wins += 1

				if error <= 5 and error >= -5:
					margin_pts += 1
					yr_margin_pts += 1
					
				if round(error) == 0:
					bonus_pts += 1
					yr_bonus_pts += 1
				
	if verbose or True:
		summary(str(yr), yr_residuals, yr_wins, yr_margin_pts, yr_bonus_pts)
		summary("ALL ", residuals, wins, margin_pts, bonus_pts)
					
	return residuals
	
def summary(name, residuals, wins, margin_pts, bonus_pts):
	
	mean = np.mean(residuals)
	stddev = np.std(residuals)
	mdm = np.abs(residuals).mean()
	mse = np.square(residuals).mean()
	winp = 100.0*wins/len(residuals)
	bru = wins+margin_pts*0.5
	
	print "%s: mean %4.1f stddev %4.1f mdm %4.1f mse %4.1f winp %3.1f m[%i] b[%i] BRU [%.1f]" % \
		(name, mean, stddev, mdm, mse, winp, margin_pts, bonus_pts, bru)

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
		help="'all' for all, 'lls' for Linear LS, '3op' for Quadratic, 'melo' for MElo")
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

	args = parser.parse_args()
	args_dict = vars(args)
	
	verbose = args_dict['v']
	data_output = args_dict['d']
	
	m = args_dict['model']
	
	if args_dict['o']:
		optimize(m)
	else:
		if m == "lls":
			runModel(LinearLSModel(), verbose, data_output)
			runModel(RoundedLLSModel(), verbose, data_output)
		elif m == "3op":
			#runModel(QuadraticModel(), verbose, data_output)
			#runModel(QuadraticModel2(), verbose, data_output)
			#runModel(RoundedQuadraticModel2(0.0), verbose, data_output)
			#runModel(RoundedQuadraticModel2(0.0), verbose, data_output)
			#runModel(RoundedQuadraticModel2(0.02), verbose, data_output)
			#runModel(RoundedQuadraticModel2(0.04), verbose, data_output)
			#runModel(RoundedQuadraticModel2(), verbose, data_output)
			#runModel(RoundedQuadraticModel3(kfactor=32), verbose, data_output)
			runModel(RoundedQuadraticModel3(kfactor=50, default_win_margin=1), verbose, data_output)
			runModel(RoundedQuadraticModel3(kfactor=50, default_win_margin=2), verbose, data_output)
			runModel(RoundedQuadraticModel3(kfactor=50, default_win_margin=3), verbose, data_output)
			runModel(RoundedQuadraticModel3(kfactor=50, default_win_margin=4), verbose, data_output)
			runModel(RoundedQuadraticModel3(kfactor=50, default_win_margin=5), verbose, data_output)
			runModel(RoundedQuadraticModel3(kfactor=50, default_win_margin=6), verbose, data_output)
			runModel(RoundedQuadraticModel3(kfactor=50, default_win_margin=9), verbose, data_output)
			runModel(RoundedQuadraticModel3(kfactor=50, default_win_margin=12), verbose, data_output)
			runModel(RoundedQuadraticModel3(kfactor=50, default_win_margin=14), verbose, data_output)
		elif m == "melo":
			runModel(MeloModel(), verbose, data_output)
		elif m == "all":
			runModel(LinearLSModel(), verbose, data_output)
			runModel(QuadraticModel(), verbose, data_output)
			runModel(RoundedQuadraticModel(), verbose, data_output)
			runModel(RoundedLLSModel(), verbose, data_output)
			runModel(MeloModel(), verbose, data_output)

if __name__ == "__main__":
	main()


# current:
"""
LS
('mean:', -0.06912423752926948)
('std dev:', 15.110670646545847)
('mean abs error:', 11.488221497058168)
('mean sq error:', 228.33714554859628)

3op
('mean:', 0.4453783010585159)
('std dev:', 15.183961311584747)
('mean abs error:', 11.484454264883649)
('mean sq error:', 230.75104294275613)
"""