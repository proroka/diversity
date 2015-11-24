function [] = plot_arena(figure_handle, task_sites, boats_pos, boats_task, make_movie)
    set(0, 'currentfigure', figure_handle); 
    clf;
    hold on;
    
    if ~make_movie
        for i = 1:size(boats_pos, 1)
            plot(boats_pos(i, :, 1), boats_pos(i, :, 2), 'b-');
        end
    end
    
    for i = 1:size(boats_pos, 1)
        plot(boats_pos(i, end, 1), boats_pos(i, end, 2), 'bo', 'MarkerFaceColor', 'b');
        %text(boats_pos(i, end, 1), boats_pos(i, end, 2), sprintf('%d', boats_task(i)), 'horizontalAlignment', 'center', 'verticalAlignment', 'middle');
    end
    plot(task_sites(:, 1), task_sites(:, 2), 'ro', 'MarkerFaceColor', 'r');
    for i = 1:size(task_sites, 1)
        %text(task_sites(i, 1), task_sites(i, 2), sprintf('%d', i), 'horizontalAlignment', 'center', 'verticalAlignment', 'middle');
    end
    %xlim([0, 3])
    %ylim([0, 3])
    axis([0 3 0 3])
    axis equal;
    grid on;
    hold off;
        
end
