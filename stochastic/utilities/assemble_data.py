# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 19:11:14 2015

@author: amandaprorok
"""

#--------
prefix = "./data/Q23/Q23_"

list_Q_a = pickle.load(open(prefix+"list_Q.p", "rb"))
t_min_mic_a = pickle.load(open(prefix+"t_min_mic.p", "rb"))
if berman:
    t_min_mic_ber_a = pickle.load(open(prefix+"t_min_mic_ber.p", "rb"))

#--------
prefix = "./data/Q24/Q24_"

list_Q_b = pickle.load(open(prefix+"list_Q.p", "rb"))
t_min_mic_b = pickle.load(open(prefix+"t_min_mic.p", "rb"))
if berman:
    t_min_mic_ber_b = pickle.load(open(prefix+"t_min_mic_ber.p", "rb"))

#--------
# combine

t_min_mic = t_min_mic_a + t_min_mic_b
t_min_mic_ber = t_min_mic_ber_a + t_min_mic_ber_b
list_Q = list_Q_a + list_Q_b

#--------
run = 'Q25'
prefix = "./data/" + run + "_"
if berman:
    pickle.dump(t_min_mic_ber, open(prefix+"t_min_mic_ber.p", "wb"))
pickle.dump(t_min_mic, open(prefix+"t_min_mic.p", "wb"))
pickle.dump(list_Q, open(prefix+"list_Q.p", "wb"))
