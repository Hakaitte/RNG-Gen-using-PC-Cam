@echo off
setlocal enabledelayedexpansion

REM Create Directory Structure

for %%d in (AlgorithmTesting BBS CCG G-SHA1 LCG MODEXP MS QCG1 QCG2 XOR) do (
    mkdir %%d\Frequency
    mkdir %%d\BlockFrequency
    mkdir %%d\Runs
    mkdir %%d\LongestRun
    mkdir %%d\Rank
    mkdir %%d\FFT
    mkdir %%d\NonOverlappingTemplate
    mkdir %%d\OverlappingTemplate
    mkdir %%d\Universal
    mkdir %%d\LinearComplexity
    mkdir %%d\Serial
    mkdir %%d\ApproximateEntropy
    mkdir %%d\CumulativeSums
    mkdir %%d\RandomExcursions
    mkdir %%d\RandomExcursionsVariant
)