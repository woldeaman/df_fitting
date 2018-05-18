% pre processing of profiles, flip and cut glass off

% first flip to have bulk on left side
flipped = transpose(fliplr(transpose(int(:, :))));

% then cut off glass, adjust cut-off bin
cut_off = 31;
cut = flipped(1:cut_off, :);

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
save('/Users/woldeaman/Desktop/data.txt', 'processed', '-ascii')
