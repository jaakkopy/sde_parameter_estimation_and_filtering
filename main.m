%% Loading the data

data = readtable('eurofxref-hist.csv');

% use data from 2020 onward
start = '2020-01-01';

r = data.USD(data.Date >= start);
dates = data.Date(data.Date >= start);

plot(dates, r);
title("EUR/USD Exchange Rate");
ylabel("Rate");
xlabel("Date");

%% Parameter estimation

function b = b_max_likelihood(r, a)
    % ML estimate for b (the long-run mean term) given a and r0, r1, ..., rn

    n = size(r,1) - 1; % -1 due to adding r0 in the first slot
    b = 1/(n*(1 - exp(-a)))*sum(r(2:end) - r(1:end-1)*exp(-a));
end

function s = s_max_likelihood(r, a)
    % ML estimate for s (the variance term) given a and r0, r1, ..., rn

    b_ml = b_max_likelihood(r, a);
    n = size(r,1) - 1;
    s = sqrt(2*a/(n*(1 - exp(-2*a)))*sum( (r(2:end) - b_ml - (r(1:end-1) - b_ml)*exp(-a)).^2 ));
end

function l = logl(r, a)
    % The likelihood of the data r0, r1, ..., rn given a, and the maximimum
    % likelihood estimates of b and s.

    n = size(r,1) - 1;

    b_ml = b_max_likelihood(r,a);
    s_ml = s_max_likelihood(r,a);

    % compute the likelihood
    SS = sum((r(2:end) - b_ml - (r(1:end-1) - b_ml) * exp(-a)).^2);
    l = -n/2 * log(s_ml^2/(2*a)) - n/2 * log(1 - exp(-2*a)) - a / (s_ml^2 * (1-exp(-2*a))) * SS;
end

r0 = r(1);
rext = [r0;r];

l = @(a) -logl(rext, a);

a_ml = fminsearch(l, 1)
b_ml = b_max_likelihood(rext, a_ml)
s_ml = s_max_likelihood(rext, a_ml)

%% Filtering

% Define the model parameters
A = exp(-a_ml);
a = b_ml*(1 - exp(-a_ml));
Q = s_ml^2*(1 - exp(-2*a_ml))/(2*a_ml);
H = 1;
R = 1e-4;
m0 = r(1);
P0 = 1;

% Predict the rates using the Kalman filter
[~, ~, mpred, Ppred] = kalman_filter(A, a, H, 0, Q, R, r', m0, P0);

Ppred = reshape(Ppred, 1, length(r));
stddev = sqrt(Ppred);

% RMSE
err = rmse(mpred, r')

% The 95% confidence bound (+- 2 standard deviations)
Y = [mpred + 2*stddev; flipud(mpred - 2*stddev)]';

% Plot the results
figure
hold on
plot(dates(2:end), r(2:end), 'Color', [0 0 1, 0.7]);
plot(dates(2:end), mpred(2:end), 'k', 'LineWidth', 1);
plot(dates(2:end), Y(2:end,:), 'k--', 'Color', [0 0 0 1]);
hold off

legend('Actual rates', 'Predicted rates', '95% Confidence interval');
xlabel('Date');
ylabel('Rate');
title('EUR/USD Exchange Rate Predictions with Kalman Filtering');