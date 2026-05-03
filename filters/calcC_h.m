function C = calcC_h(x)
%% Analytical Jacobian of func_h: C = dh/dx  (3x16)
% h(x) = [v_bx; v_by; pd]  where v_b = R_n2b * v_ned
% Used by EKF_UAV and REKF_UAV for the flow+baro measurement model.

q0=x(1); q1=x(2); q2=x(3); q3=x(4);
vn=x(5); ve=x(6); vd=x(7);

C = zeros(3,16);

% d(v_bx)/dq  and  d(v_bx)/dv_ned
C(1,1:4) = 2*[q0*vn+q3*ve-q2*vd,  q1*vn+q2*ve+q3*vd, -q2*vn+q1*ve-q0*vd, -q3*vn+q0*ve+q1*vd];
C(1,5:7) = [q0^2+q1^2-q2^2-q3^2,  2*(q1*q2+q0*q3),    2*(q1*q3-q0*q2)];

% d(v_by)/dq  and  d(v_by)/dv_ned
C(2,1:4) = 2*[-q3*vn+q0*ve+q1*vd,  q2*vn-q1*ve+q0*vd,  q1*vn+q2*ve+q3*vd, -q0*vn-q3*ve+q2*vd];
C(2,5:7) = [2*(q1*q2-q0*q3),        q0^2-q1^2+q2^2-q3^2, 2*(q2*q3+q0*q1)];

% d(pd)/d(pd)
C(3,10) = 1;
end
