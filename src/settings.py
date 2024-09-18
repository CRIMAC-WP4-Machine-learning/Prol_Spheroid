

class LiquidFilledSettings:

    def __init__(self):
        self.prefix = 'liquid'
        # media properties (w:media surrounding the spheroid, s:spheroid)
        self.ro_w = 1027
        self.ro_s = 1027 * 1.05
        self.c_w = 1500
        self.c_s = 1500 * 1.05

        # geometrical properties
        self.a = 0.08
        self.b = 0.02

        # frequencies
        self.delta_f = 2000
        self.min_freq = 100000
        self.max_freq = 120100

        #incidence angle
        self.theta_i_deg = 90


class AirFilledSettings:

    def __init__(self):
        self.prefix = 'airfilled'
        # media properties (w:media surrounding the spheroid, s:spheroid)
        self.ro_w = 1027
        self.ro_s = 10
        self.c_w = 1500
        self.c_s = 343

        # geometrical properties
        self.a = 0.03
        self.b = 0.01

        # frequencies
        self.delta_f = 1000
        #self.min_freq = 269000
        #self.max_freq = 273100
        #self.min_freq = 100000
        #self.max_freq = 300100
        self.min_freq = 38000
        self.max_freq = 300100

        #incidence angle
        self.theta_i_deg = 90
