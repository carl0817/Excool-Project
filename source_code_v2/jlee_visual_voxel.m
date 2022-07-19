[W, Rmat, Lambda] = voxelwise_encoding(Y, X, lambda, 9)
[W, Rmat, Errmat, Lambda] = visual_reconstruction(fmri_avg',  lay_feat_concatenated, linspace(0,0.1,100) , 9, [])