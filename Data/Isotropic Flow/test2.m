

load('Downsampled_PIV_R4_T1-20_Int1.mat-PY.mat');   % 修改成你的文件名

u = u_remain(:);
v = v_remain(:);
w = w_remain(:);

fprintf("u: min = %.4f, max = %.4f\n", min(u), max(u));
fprintf("v: min = %.4f, max = %.4f\n", min(v), max(v));
fprintf("w: min = %.4f, max = %.4f\n", min(w), max(w));


edges = linspace(min([u;v;w]), max([u;v;w]), 50);  % 50 个区间，可自己改
[count_u, ~] = histcounts(u, edges);
[count_v, ~] = histcounts(v, edges);
[count_w, ~] = histcounts(w, edges);


figure;
plot(edges(1:end-1), count_u, '-r', 'LineWidth', 1.4); hold on;
plot(edges(1:end-1), count_v, '-g', 'LineWidth', 1.4);
plot(edges(1:end-1), count_w, '-b', 'LineWidth', 1.4);
grid on; xlabel('Value'); ylabel('Count');
legend('u', 'v', 'w');
title('Value Distribution (bin counts)');


figure;
subplot(1,3,1);
histogram(u, 100); grid on; title('u distribution');

subplot(1,3,2);
histogram(v, 100); grid on; title('v distribution');

subplot(1,3,3);
histogram(w, 100); grid on; title('w distribution');
