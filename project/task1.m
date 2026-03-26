function y = optical_flow_model(x)
    %state x = [q0, q1, q2, q3, vn, ve, vd, pn, pe, pd, wb_x, wb_y, wb_z,
    %ab_x, ab_y, ab_z]
    q = x(1:4);
    v_n = x(5:7);
    p_d = x(10);

    R_n2b = quat2rotm(q)';

    v_b = R_n2b * v_n;

    y = [v_b(1);
        v_b(2);
        v_n(3);
        p_d];
end