function output = calculate_hdrvdp3(HQ_img_path, GT_img_path)

HQ_img = imread(HQ_img_path);
GT_img = imread(GT_img_path);
HQ_img = im2double(HQ_img);
GT_img = im2double(GT_img);
HQ_img = PQEOTF(HQ_img);
GT_img = PQEOTF(GT_img);

ppd = 50;
res = hdrvdp3('side-by-side', HQ_img, GT_img, 'rgb-bt.2020', ppd, {'rgb_display', 'led-lcd-wcg', 'disable_lowvals_warning', true});

output = res.Q;

end