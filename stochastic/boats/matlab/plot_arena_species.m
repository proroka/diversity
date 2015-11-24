function [] = plot_arena_species(figure_handle, task_sites, boats_pos, boats_task, boats_species, make_movie)
    set(0, 'currentfigure', figure_handle); 
    clf;
    hold on;
    
    col_l = {'m-','g-','c-','b-'};
    col_o = {'mo','go','co','bo'};
    col = {'m','g','c','b'};

    nboats = size(boats_pos, 1);
    
    if ~make_movie
        for i = 1:nboats
            plot(boats_pos(i, :, 1), boats_pos(i, :, 2), col_l{boats_species(i)});
        end
    end
    
    for i = 1:size(boats_pos, 1)
        plot(boats_pos(i, end, 1), boats_pos(i, end, 2), col_o{boats_species(i)}, 'MarkerFaceColor', col{boats_species(i)});
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
