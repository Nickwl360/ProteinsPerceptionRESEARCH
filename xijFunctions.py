from SCDcalc import *
import math


def calcSDij(seq, i, j):
    N = len(seq)
    q = getcharges(seq)

    s1 = 0

    for m in range(j, i + 1):
        nloop = 0
        for n in range(1, j):
            nloop += q[m - 1] * q[n - 1] * (m - j) ** (2) / ((m - n) ** (3 / 2))
        s1 += nloop

    s2 = 0

    for m in range(j + 1, i + 1):
        nloop = 0
        for n in range(j, m):
            nloop += q[m - 1] * q[n - 1] * (m - n) ** (1 / 2)
        s2 += nloop

    s3 = 0

    for m in range(i + 1, N + 1):
        nloop = 0
        for n in range(1, j):
            nloop += q[m - 1] * q[n - 1] * (i - j) ** 2 / ((m - n) ** (3 / 2))
        s3 += nloop

    s4 = 0

    for m in range(i + 1, N + 1):
        nloop = 0
        for n in range(j, i + 1):
            nloop += q[m - 1] * q[n - 1] * (i - n) ** (2) / ((m - n) ** (3 / 2))
        s4 += nloop

    return ((s1 + s2 + s3 + s4) / (i - j))


def calcSHij(seq, i, j):
    N = len(seq)

    s1 = 0
    for m in range(j, i + 1):
        nloop = 0
        for n in range(1, j):
            nloop += (m - j) ** (2) / ((m - n) ** (5 / 2))
        s1 += nloop

    s2 = 0
    for m in range(j + 1, i + 1):
        nloop = 0
        for n in range(j, m):
            nloop += (m - n) ** (-0.5)
        s2 += nloop

    s3 = 0
    for m in range(i + 1, N + 1):
        nloop = 0
        for n in range(1, j):
            nloop += (i - j) ** 2 / ((m - n) ** (5 / 2))
        s3 += nloop

    s4 = 0
    for m in range(i + 1, N + 1):
        nloop = 0
        for n in range(j, i + 1):
            nloop += (i - n) ** (2) / ((m - n) ** (5 / 2))
        s4 += nloop

    return ((s1 + s2 + s3 + s4) / (i - j))


def calcTij(seq, i, j):
    N = len(seq)

    s1 = 0
    for l in range(j + 1, i + 1):
        mloop = 0
        for m in range(2, j + 1):
            nloop = 0
            for n in range(1, m):
                nloop += (l - j) ** 2 / ((l - m) ** (5 / 2) * (m - n) ** (3 / 2))
            mloop += nloop
        s1 += mloop

    s2 = 0
    for l in range(j + 2, i + 1):
        mloop = 0
        for m in range(j + 1, l):
            nloop = 0
            for n in range(1, j + 1):
                nloop += ((l - m) ** 2 / ((l - m) ** (5 / 2) * (m - n) ** (3 / 2)) + (m - j) ** 2 / (
                        (l - m) ** (3 / 2) * (m - n) ** (5 / 2)))
            mloop += nloop
        s2 += mloop

    s3 = 0
    for l in range(i + 1, N + 1):
        mloop = 0
        for m in range(2, j + 1):
            nloop = 0
            for n in range(1, m):
                nloop += (i - j) ** 2 / ((l - m) ** (5 / 2) * (m - n) ** (3 / 2))
            mloop += nloop
        s3 += mloop

    s4 = 0
    for l in range(i + 1, N + 1):
        mloop = 0
        for m in range(i, l):
            nloop = 0
            for n in range(1, j + 1):
                nloop += (i - j) ** 2 / ((l - m) ** (3 / 2) * (m - n) ** (5 / 2))
            mloop += nloop
        s4 += mloop

    s5 = 0
    for l in range(i + 1, N + 1):
        mloop = 0
        for m in range(j + 1, i):
            nloop = 0
            for n in range(1, j + 1):
                nloop += ((i - m) ** 2 / ((l - m) ** (5 / 2) * (m - n) ** (3 / 2)) + (m - j) ** 2 / (
                        (l - m) ** (3 / 2) * (m - n) ** (5 / 2)))
            mloop += nloop
        s5 += mloop

    s6 = 0
    for l in range(j + 3, i + 1):
        mloop = 0
        for m in range(j + 2, l):
            nloop = 0
            for n in range(j + 1, m):
                nloop += ((l - m) ** 2 / ((l - m) ** (5 / 2) * (m - n) ** (3 / 2)) + (m - n) ** 2 / (
                        (l - m) ** (3 / 2) * (m - n) ** (5 / 2)))
            mloop += nloop
        s6 += mloop

    s7 = 0
    for l in range(i + 1, N + 1):
        mloop = 0
        for m in range(j + 2, i + 1):
            nloop = 0
            for n in range(j + 1, m):
                nloop += ((i - m) ** 2 / ((l - m) ** (5 / 2) * (m - n) ** (3 / 2)) + (m - n) ** 2 / (
                        (l - m) ** (3 / 2) * (m - n) ** (5 / 2)))
            mloop += nloop
        s7 += mloop

    s8 = 0
    for l in range(i + 2, N + 1):
        mloop = 0
        for m in range(i + 1, l):
            nloop = 0
            for n in range(j + 1, i + 1):
                nloop += (i - n) ** 2 / ((l - m) ** (3 / 2) * (m - n) ** (5 / 2))
            mloop += nloop
        s8 += mloop

    return ((s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8) / (i - j))
