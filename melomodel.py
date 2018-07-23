#=========================================================================
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