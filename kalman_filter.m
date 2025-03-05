function [mest, Pest, mpred, Ppred] = kalman_filter(A, a, H, b, Q, R, Y, m0, P0)
%KALMAN_FILTER Computes filtered state estimates assuming the model
% x(t+1) = Ax(t) + a + q(t),  q ~ N(0,Q)
% y(t+1) = Hx(t+1) + b + r(t),  r ~ N(0,R)
% INPUT
% A: NxN state transition matrix
% a: Nx1 constant term
% Q: NxN process noise covariance matrix
% H: MxN measurement model matrix
% b: Mx1 constant term
% R: MxM measurement noise covariance matrix
% Y: MxT array of measurement observations
% m0: Nx1 prior distribution mean
% P0: NxN prior distribution covariance
% OUTPUT
% mest: NxT array of state estimates
% Pest: NxNxT array of covariance estimates
% mpred: NxT array of state predictions
% Ppred: NxNxT array of covariance predictions

% The implementation is based on the version presented by Simo Särkkä and
% Lennart Svensson, Bayesian Filtering and Smoothing, 2nd ed. (2013)

N = size(A,1);
T = size(Y,2);

mest = zeros(N,T);
mpred = zeros(N,T);
Pest = zeros(N,N,T);
Ppred = zeros(N,N,T);

m = m0;
P = P0;

for k = 1:T
   % Prediction step
   m_ = A*m + a;
   P_ = A*P*A' + Q;

   % Update step
   v = Y(:,k) - H*m_ - b;
   S = H*P_*H' + R;
   K = P_*H'/S;
   m = m_ + K*v;
   P = P_ - K*S*K';
   
   % Store the results of this iteration
   mest(:,k) = m;
   mpred(:,k) = m_;
   Pest(:,:,k) = P;
   Ppred(:,:,k) = P_;
end

end

