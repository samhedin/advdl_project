"""
This module contains the code to compute different metrics for generated images
- Inception Score
- FID

## Inception Score
- Use pretrained Inception V3 to predict classes for each generated image
- The entropy is calculated as the negative sum of each observed probability multiplied by the log of the probability.
  entropy = -sum(p_i * log(p_i))
- 
"""