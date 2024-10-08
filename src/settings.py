

class LiquidFilledSettings:

    def __init__(self):
        self.prefix = 'liquid'
        # media properties (w:media surrounding the spheroid, s:spheroid)
        self.ro_w = 1027
        self.ro_s = 1027 * 0.25
        self.c_w = 1500
        self.c_s = 1500 * 0.25

        # geometrical properties
        self.a = 0.1
        self.b = 0.005

        # frequencies
        self.delta_f = 1000
        self.min_freq = 75000
        self.max_freq = 75100

        #incident angle
        self.theta_i_deg = 15

        #precision
        self.precision_fbs = 1e-6


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

        # incident angle
        self.theta_i_deg = 90

        # precision
        self.precision_fbs = 1e-5
