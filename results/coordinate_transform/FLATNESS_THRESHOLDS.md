# Coordinate-transform flatness threshold candidates

The source measurements are in `flatness_threshold_data.csv`. Negative outcome percentages mean that enabling the coordinate transform improves compression.

## Stage 1: before training

- Force enable when `original_flatness<=0.05`.
- Force reject when `original_flatness>=0.75`.
- Train the coordinate transform whenever `original_flatness<0.75`; the low-flatness region is enabled without a second-stage decision.

The enable threshold currently selects only Katrina (`0.005934`),where the current explicit comparison improves BD-rate by `49.528%`. The reject threshold covers E3SM and ADCIRC Haiyan. It does not miss any measured improvement above `2%`,but it deliberately gives up three small E3SM improvements of `0.709%`,`0.774%`,and`1.677%` to avoid training and prevent the larger regressions observed on other E3SM fields.

## Stage 2: relative flatness change

- Enable when `transformed_flatness/original_flatness>=1.075`.
- Otherwise disable the transform.

Together with stage 1,this enables 11 of the 32 measured fields,with the same three small E3SM exceptions and no harmful enable decision above `0.5%`.

This relative rule is the final automatic decision. It does not fall back to the old loss-gain threshold,which is known to enable the severely harmful Tpost20 transform.

Flatness can be computed from the coordinate covariance accumulated during the existing coordinate scan. After training,a diagonal transform updates the covariance as `D*C*D`,so transformed flatness requires only a small eigenvalue calculation and no second coordinate pass.
