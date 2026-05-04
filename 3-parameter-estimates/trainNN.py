# TODO:
# Test mini batches with fresh simulated data (try to implement fast data generation in Python)
# Implement 1D CNN on log(y^2 + k) data which encodes into a shared trunk
# Test out ARMA(1, 1) coefficients in summary_stats()
# If mini batches is a success, do a mathematical analysis in the thesis to justify approach
# Clean up in the scaling and standarization nightmare
# Do some prompt engineering to speed up the briefing phase of GPT


# When creating a joint posterior, assume independence
# Prioritize runtime efficiency and simplicity as we need a baseline model first
#   