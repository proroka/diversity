function dx = compute_velocity(x, radius, center, obstacles, velocity_on_circle, avoidance_velocity, avoidance_range)
    if nargin < 4
        obstacles = [];
    end
    if nargin < 5
        velocity_on_circle = 0.2;
    end
    if nargin < 6
        avoidance_velocity = 0.1;
    end
    if nargin < 7
        avoidance_range = 0.5;
    end


    dx = zeros(2, 1);
    tx = x;
    tx(1) = tx(1) - center(1);
    tx(2) = tx(2) - center(2);
    dx(1) =   tx(1) + tx(2) - tx(1) * (tx(1)^2 + tx(2)^2) / (radius * radius);
    dx(2) = - tx(1) + tx(2) - tx(2) * (tx(1)^2 + tx(2)^2) / (radius * radius);
    dx = dx * velocity_on_circle;

    % Add repulsive forces for obstacle avoidance.
    % Only when obstacles are closer than avoidance_range apart.
    for i = 1:size(obstacles, 1)
        pos = obstacles(i, :).';
        dpos = (x - pos);
        dist = norm(dpos);
        if dist < avoidance_range
            dpos = dpos / dist;  % normalize.
            max_value = normpdf(0, 0, 3 * avoidance_range);
            dx = dx + dpos * avoidance_velocity * normpdf(dist, 0, 3 * avoidance_range) / max_value;
        end
    end
end
