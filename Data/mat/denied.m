function [y] = denied(y, T, I, Detla)
% T: Start time of denial
% I: Duration of the denial
% Detla: Data frequency (Here is w.r.p to IMU)

% GPS denied data
N = size(y, 1);
for k = T*Detla+1 : min((T+I)*Detla, N)
    y(k,:) = y(T*Detla,:);
end

end