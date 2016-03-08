% File Type:     Matlab
% Author:        Hanxi Li {lihanxi2001@gmail.com}
% Creation:      Saturday 30/01/2016 18:00.
% Last Revision: Saturday 30/01/2016 18:00.

function HanxiShowboxes(im, boxes, legends, line_style)

    % Draw bounding boxes on top of an image.
    %   showboxes(im, boxes)
    %
    % -------------------------------------------------------

    fix_width = 1200;
    if isa(im, 'gpuArray')
        im = gather(im);
    end
    imsz = size(im);
    scale = fix_width / imsz(2);
    im = imresize(im, scale);

    if size(boxes{1}, 2) >= 5
        boxes = cellfun(@(x) ...
            [x(:, 1:4) * scale, x(:, 5)], boxes, 'UniformOutput', false);
    else
        boxes = cellfun(@(x) ...
            x(:, 1:4) * scale, boxes, 'UniformOutput', false);
    end

    imagesc(im);axis equal; axis off;
    set(gcf, 'menubar', 'none');
    set(gcf, 'Color', 'white');hold on;
    set(gca,'position',[0.1 0.1 0.8 0.8],'units','normalized')

    num_class = length(boxes);
    valid_boxes = cellfun(@(x) ~isempty(x), boxes, 'UniformOutput', true);
    if sum(valid_boxes) == 0, return; end

    colors = assign_class_color(num_class);
    for i_class = 1 : num_class

        if isempty(boxes{i_class}), continue; end

        line_style_now = line_style{i_class};
        for i_box = 1:size(boxes{i_class})
            box = boxes{i_class}(i_box, 1 : 4);
            if size(boxes{i_class}, 2) >= 5
                score = boxes{i_class}(i_box, 5);
                linewidth = 0.5 + min(max(score, 0), 1) * 2.5;
                rectangle('Position', RectLTRB2LTWH(box), 'LineWidth', ...
                            linewidth, 'EdgeColor', colors{i_class}, ...
                                    'LineStyle', line_style_now{i_box});
%                 label = sprintf('%s : %.3f', legends{i_class}, score);
%                 text(double(box(1))+2, double(box(2)), label, 'BackgroundColor', 'w');
            else
                linewidth = 2;
                rectangle('Position', RectLTRB2LTWH(box), 'LineWidth', ...
                            linewidth, 'EdgeColor', colors{i_class}, ...
                                            'LineStyle', line_style_now{i_box});
%                 label = sprintf('%s(%d)', legends{i_class}, i_class);
%                 text(double(box(1))+2, double(box(2)), label, 'BackgroundColor', 'w');
            end
        end

    end

end

%% ************************************************************************ %%
function colors = assign_class_color(num_class)
    colors = colormap('hsv');
    colors = colors(1 : (floor(size(colors, 1) / num_class)) : end, :);
    colors = mat2cell(colors, ones(size(colors, 1), 1))';
end

%% ************************************************************************ %%
function [ rectsLTWH ] = RectLTRB2LTWH( rectsLTRB )
    %rects (l, t, r, b) to (l, t, w, h)
    rectsLTWH = [rectsLTRB(:, 1), rectsLTRB(:, 2), ...
        rectsLTRB(:, 3)-rectsLTRB(:,1)+1, rectsLTRB(:,4)-rectsLTRB(2)+1];
end



