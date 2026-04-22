function [x_pred, P_pred] = UKF(x_0, y, P0, B, D, f, h, sym_x, len)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robust UKF 
%
% Input:
%       f/h: Model
%       x_0: Initial condition x(1|0)
%       P0 : Initial condition variance P(1|0)
%       B  : Matrix B of the noise inputs
%       D  : Matrix D of the noise inputs
%       y  : Measured data
%       len: Length of the dataset, and so the length of the iterations

% Outputs:
%       x_pred: state predcited   x(t+1|t)
%       P_pred: state performance P(t+1|t)

%% SAVE
Q = B*B';
R = D*D';
n = size(P0,1);
m = size(R,1);
x_pred = zeros(n,len+1);  %save \hat x_{t}
x_hat  = zeros(n,len);  %save \hat x_{t|t}
y_pred = zeros(m,len);  %save \hat y_{t}
P_pred = zeros(n,n,len+1); %save P_t
Wc     = zeros(1,2*n+1); %save weights Wc
Wm     = zeros(1,2*n+1); %save weights Wm

%% Scaling factor (see Wan and Van der Merwe, 2001)
alpha = 0.1; % small postive value (10^-4 ~ 1)
kapa  = 3-n; 
lambda= alpha^2*(kapa+n)-n;
beta  = 2;   % for Gaussian 2 is optimal

%% Initial
% First prediction (i.e. initial conditions) x(1|0)
x_pred(:,1) = x_0; 

% First prediction variance P0 = P(1|0)
P_pred(:,:,1) = P0;

%% Iterative part
for t = 1:len
    %% SAVE
    X      = zeros(n,2*n+1); %save Sigma points for X^(i)_t
    Y      = zeros(m,2*n+1); %save Sigma points for Y^(i)_t=h(X^(i)_t)
    X_hat  = zeros(n,2*n+1); %save Sigma points for X^(i)_t|t
    X_pred = zeros(n,2*n+1); %save Sigma points for X_pred^(i)_t+1=f(X^(i)_t|t)
%% Generate sigma points of x_pred_t
    for i = 1:n
        sqrtP   = chol(P_pred(:,:,t)); 
        X(:,i)  = x_pred(:,t) + sqrt(n+lambda)*sqrtP(i,:)';
        X(:,i+n)= x_pred(:,t) - sqrt(n+lambda)*sqrtP(i,:)';
    end
    X(:,2*n+1)  = x_pred(:,t); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Weights Wc/Wm
    Wc(1:2*n) = 1/(2*(n+lambda));
    Wc(2*n+1) = lambda/(n+lambda) + 1 - alpha^2 + beta;
    Wm(1:2*n) = 1/(2*(n+lambda));
    Wm(2*n+1) = lambda/(n+lambda);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%% Compute Y^(i)_t=h(X^(i)_t) and its mean
    for i=1:2*n+1
        Y(:,i)      = subs(h,sym_x,X(:,i));
        y_pred(:,t) = y_pred(:,t)+Wm(i)*Y(:,i);
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%% Find the Kalman estimation gain
    P_y  = zeros(m,m); %sum^{2n}_0 {W^i*(Y^i-mean_Y)(Y^i-mean_Y)^T+R};
    P_xy = zeros(n,m); %sum^{2n}_0 {W^i*(X^i-mean_x)(Y^i-mean_Y)^T};
    for i=1:2*n+1
        P_y =P_y +Wc(i)*(Y(:,i)-y_pred(:,t))*(Y(:,i)-y_pred(:,t))';
        P_xy=P_xy+Wc(i)*(X(:,i)-x_pred(:,t))*(Y(:,i)-y_pred(:,t))';
    end
    P_y = P_y + R;
    L   = P_xy*inv(P_y);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Compute x(t|t)
    x_hat(:,t) = x_pred(:,t) + L*(y(t)-y_pred(:,t)); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Compute P(t|t)
    P_hat = P_pred(:,:,t) - L*P_y*L';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Generate Sigma points of x_hat_t
    for i = 1:n
        sqrtP_hat   = chol(P_hat); 
        X_hat(:,i)  = x_hat(:,t) + sqrt(n+lambda)*sqrtP_hat(i,:)';
        X_hat(:,i+n)= x_hat(:,t) - sqrt(n+lambda)*sqrtP_hat(i,:)';
    end
    X_hat(:,2*n+1)  = x_hat(:,t); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Compute X^(i)_t+1=f(X_hat^(i)_t) and its mean
    for i=1:2*n+1
        X_pred(:,i)   = subs(f,sym_x,X_hat(:,i));
        x_pred(:,t+1) = x_pred(:,t+1)+Wm(i)*X_pred(:,i);
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Compute the prediction variance P(t+1|t)
    for i=1:2*n+1
        P_pred(:,:,t+1)=P_pred(:,:,t+1)+Wc(i)...
            *(X_pred(:,i)-x_pred(:,t+1))*(X_pred(:,i)-x_pred(:,t+1))';
    end
    P_pred(:,:,t+1) = P_pred(:,:,t+1)+Q;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end