% pre processing of profiles, flip and cut glass off

name = 'gel6_dex70_forts'  % supply name for profiles
% first flip to have bulk on left side
flipped = transpose(fliplr(transpose(int(:, :))));

% then cut off glass, adjust cut-off bin here
cut_off = 40;
cut = flipped(1:cut_off, :);
cut = cut./cut(1);  % set concentration to one at leftmost bin in the beginning

% create x-vector
dx = 10;
xx = linspace(0, (length(cut(:, 1)) - 1)*dx, length(cut(:, 1)));

% check if everything worked as expected
plot(xx, cut, 'o');

% save to new matrix
processed = [];
processed(:, 1) = xx;
processed(:, 2:length(cut)+1) = cut;

% save to file
dlmwrite(strcat('/Users/woldeaman/Desktop/', name, '.txt'), processed, 'delimiter', ',');
