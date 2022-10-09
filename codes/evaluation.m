pd_folder = '../experiments/fmnet_final/val_images'; 
gt_folder = '../data/hdrtv1k/test_set/test_hdr';

cd('./hdr_toolbox');
installHDRToolbox

cd('..');
addpath('./hdrvdp-3.0.6/');

pd_files = dir(strcat(pd_folder, '/*.png'));

hdrvdp3_score_all = 0;
srsim_score_all = 0;

for i = 1:length(pd_files)
    pd_file = strcat(pd_folder, '/', pd_files(i).name);
    gt_file = strcat(gt_folder, '/', pd_files(i).name);
    fprintf('Processing: %s\n', pd_file);
    hdrvdp3_score = calculate_hdrvdp3(pd_file, gt_file);
    srsim_score = calculate_srsim(pd_file, gt_file);
    hdrvdp3_score_all = hdrvdp3_score_all + hdrvdp3_score;
    srsim_score_all = srsim_score_all + srsim_score;
end

fprintf('HDRVDP3: %f\n', hdrvdp3_score_all/length(pd_files));
fprintf('SR-SIM: %f\n', srsim_score_all/length(pd_files));