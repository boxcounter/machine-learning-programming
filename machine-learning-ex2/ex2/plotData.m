function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

positive = [];
negative = [];

for i = 1:length(y)
    switch (y(i))
        case 1
            positive = [positive; X(i,:)];
        case 0
            negative = [negative; X(i,:)];
    end
end

scatter(positive(:,1), positive(:,2), [], 'k', '+', 'filled', 'markeredgecolor', 'k', 'linewidth', 2);
scatter(negative(:,1), negative(:,2), [], 'y', 'o', 'filled', 'markeredgecolor', 'k');

% =========================================================================



hold off;

end
