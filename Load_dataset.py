import numpy as np
import pandas as pd
import nibabel as nib
import argparse
import time
import os 
import logging
from keras.models import Model
from keras.layers import Dense, Input
from sklearn.model_selection import train_test_split

np.random.seed(123)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.system("mkdir Process_data")

parser = argparse.ArgumentParser()
parser.add_argument('--AllenProbes', type=str, default='./Raw_data/Allen_Human_Brain_Atlas/AllProbes.csv')
parser.add_argument('--AllenExpr', type=str, default='./Raw_data/Allen_Human_Brain_Atlas/AllMicroarrayExpression.csv')
parser.add_argument('--AllenSample', type=str, default='./Raw_data/Allen_Human_Brain_Atlas/AllSampleAnnot.csv')
parser.add_argument('--fMRICIFTI', type=str, default='./Raw_data/Human_Connectome_Project/HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500_Eigenmaps.dtseries.nii')
parser.add_argument('--parcelCIFTI', type=str, default='./Raw_data/CABNP_framework/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii')
parser.add_argument('--parcelTSF', type=str, default='./Raw_data/CABNP_framework/Output_Atlas_CortSubcort.Parcels.LR.ptseries.nii')
parser.add_argument('--parcelKEY', type=str, default='./Raw_data/CABNP_framework/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt')
parser.add_argument('--rsurf', type=str, default='./Raw_data/CABNP_framework/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii')
parser.add_argument('--lsurf', type=str, default='./Raw_data/CABNP_framework/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii')
parser.add_argument('--GRNdb', type=str, default='./Raw_data/GRNdb/Brain_GTEx-regulons.txt')
parser.add_argument('--GTEx', type=str, default='./Raw_data/GTEx/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct')
parser.add_argument('--GTExSample', type=str, default='./Raw_data/GTEx/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt')
parser.add_argument('--genehistory', type=str, default='./Raw_data/GRCh38.p13/gene_history.gz')
parser.add_argument('--gene2ensembl', type=str, default='./Raw_data/GRCh38.p13/gene2ensembl.gz')
parser.add_argument('--HumanRNA1', type=str, default='./Raw_data/GRCh38.p13/GCF_000001405.39_GRCh38.p13_rna.fna')
parser.add_argument('--HumanRNA2', type=str, default='./Raw_data/GRCh38.p13/Additional.rna.fna')
parser.add_argument('--HumanPEP1', type=str, default='./Raw_data/GRCh38.p13/GCF_000001405.39_GRCh38.p13_protein.faa')
parser.add_argument('--HumanPEP2', type=str, default='./Raw_data/GRCh38.p13/Additional.pep.fna')
parser.add_argument('--GeneDisease', type=str, default='./Raw_data/DisGeNet/all_gene_disease_associations.tsv')
parser.add_argument('--VariantDisease', type=str, default='./Raw_data/DisGeNet/all_variant_disease_associations.tsv')
parser.add_argument('--enhanceMT', type=str, default='enhance_matrix.txt')
parser.add_argument('--makeblastdb', type=str, default='/public/home/HYP_liaol/tools/ncbi-blast-2.11.0+/bin/makeblastdb')
parser.add_argument('--blastp', type=str, default='/public/home/HYP_liaol/tools/ncbi-blast-2.11.0+/bin/blastp')
parser.add_argument('--wb', type=str, default='/public/home/HYP_liaol/tools/workbench/bin_rh_linux64/wb_command')

#parser.add_argument('--makeblastdb', type=str, default='/Applications/ncbi-blast-2.13.0+/bin/makeblastdb')
#parser.add_argument('--blastp', type=str, default='/Applications/ncbi-blast-2.13.0+/bin/blastp')
#parser.add_argument('--wb', type=str, default='/Applications/workbench/bin_macosx64/wb_command')

parser.add_argument('--tmp', type=str, default='./Process_data')
args = parser.parse_args(args = [])

def LocalTime():
    return(time.strftime(" %Y-%m-%d %H:%M:%S\n --> ", time.localtime()))

def Get_GR():
    print(LocalTime(), "Start getting GR network...")
    # Import 58692 probe informations
    probe_info_df = pd.read_csv(args.AllenProbes)
    probe_name_list = probe_info_df["probe_name"].dropna(axis = 0).drop_duplicates().tolist()
    gene_entrez_list = probe_info_df["entrez_id"].dropna(axis = 0).drop_duplicates().tolist()
    print(LocalTime(), "Find",len(probe_name_list),"cover",len(gene_entrez_list),"unique genes...")
    
    # Remove probes with NA genes
    probe_info_df = probe_info_df.dropna(axis=0, how='any').reset_index(drop = True)
    probe_info_df["entrez_id"] = "E"+probe_info_df["entrez_id"].apply(int).apply(str)
    probe_name_list = probe_info_df["probe_name"].dropna(axis = 0).drop_duplicates().tolist()
    gene_entrez_list = probe_info_df["entrez_id"].dropna(axis = 0).drop_duplicates().tolist()
    
    # Create a zero matrix, 1 = this probe related this gene
    gene_probe_matrix = np.zeros((len(gene_entrez_list), len(probe_name_list)), dtype=int) 
    gene_probe_df = pd.DataFrame(gene_probe_matrix, index = gene_entrez_list, columns = probe_name_list) 
    print(LocalTime(), "Matching probes and related unique genes...")
    for i in probe_name_list:
        i_probe_name = i
        i_entrez = probe_info_df["entrez_id"][probe_info_df["probe_name"] == i].tolist()[0]
        gene_probe_df.loc[i_entrez, i_probe_name] = 1
        
    print(LocalTime(), "Loading probe expression values in brain regions...")
    # Create a dict stored probe id and name for match brain region
    probe_id_name_dict = dict(zip(probe_info_df["probe_id"].tolist(), probe_info_df["probe_name"].tolist()))
    
    # Import 58692 probe expression values in 3702 brain regions (58692*3702)
    probe_expr_df = pd.read_csv(args.AllenExpr, header = None, index_col = 0)
    
    # Only keep the probes which cover genes
    probe_expr_df = probe_expr_df.loc[probe_id_name_dict.keys(),:]
    
    # Replace probe id to probe name as index
    probe_expr_df.rename(index = probe_id_name_dict, inplace=True)
    probe_expr_matrix = probe_expr_df.values
    
    print(LocalTime(), "Get expression levels of",len(gene_entrez_list), "unique genes in",probe_expr_df.shape[1], "brain regions...")
    gene_region_expr_matrix = np.dot(gene_probe_df.values, probe_expr_matrix)
    
    # However, one gene may related to two or more probes, hence the expression values
    # in some regions may be overestimated. Here, I make the expression value of each gene
    # represent by single probe.
    gene_region_mean_expr_matrix = (gene_region_expr_matrix.T / gene_probe_df.values.sum(axis = 1)).T
    # Normalize the expression
    normalized_expr_matrix = (gene_region_mean_expr_matrix - np.mean(gene_region_mean_expr_matrix, axis = 1, keepdims = True)) / (np.std(gene_region_mean_expr_matrix, axis = 1, keepdims = True) + 1e-8)
    
    # Load region information
    sample_info_df = pd.read_csv(args.AllenSample, sep = ",")
    sample_info_df_raw_index = list(map(str, sample_info_df.index.tolist()))
    sample_info_df_new_index = ["Region_" + x for x in sample_info_df_raw_index]
    sample_info_df.index = sample_info_df_new_index
    sample_info_df = sample_info_df.rename_axis("region_name", axis = 0)
    
    normalized_expr_df = pd.DataFrame(normalized_expr_matrix, index = gene_probe_df.index.tolist(), columns=sample_info_df.index.tolist())
    sample_info_df.rename_axis(index=None, columns=None, inplace=True)
    return(normalized_expr_df, sample_info_df)
    # Return gene-region matrix, and regions informations

def Get_PP():
    # Load in dense array time series using nibabel
    # shape: (4500, 91282)
    # ~4500 time points, ~90000 voxels (~60000 surface voxels and ~30000 subcortical and cerebellar voxels)
    print(LocalTime(), "Start getting PP network...")
    print(LocalTime(), "Loading r-fMRI data...")
    dtseries = np.squeeze(nib.load(args.fMRICIFTI).get_fdata())
    
    # Parcellate dense time series using wb_command
    print(LocalTime(), "Partition r-fMRI data to 718 parcels...")
    os.system(args.wb + ' -cifti-parcellate ' + args.fMRICIFTI + ' ' + args.parcelCIFTI + ' COLUMN ' + args.parcelTSF + ' -method MEAN')
    
    # Load in parcellated data using nibabel
    # 718 parcels * 4500 timepoints
    lr_parcellated = np.squeeze(nib.load(args.parcelTSF).get_fdata()).T
    
    # Computing functional connectivity and visualizing the data (assuming preprocessing has already been done)
    FCmat = np.corrcoef(lr_parcellated)
    
    # Load parcel information
    parcel_info = pd.read_table(args.parcelKEY, sep  = "\t", index_col = "INDEX")
    
    # FCmat order is same as parcel_info now, we will order them by NETWORKSORTEDORDER
    parcel_info_sort = parcel_info.sort_values(by = "NETWORKSORTEDORDER")
    parcel_order = np.array(parcel_info_sort.index - 1)
    parcel_order.shape = (len(parcel_order), 1)
    
    # 718*718 region connectively network
    print(LocalTime(), "Compute functional connectively between parcels")
    FCMat_sorted = FCmat[parcel_order, parcel_order.T]
    
    FCMat_df = pd.DataFrame(FCMat_sorted, index = parcel_info_sort["LABEL"], columns = parcel_info_sort["LABEL"])
    FCMat_df_info = parcel_info_sort[["LABEL","KEYVALUE","HEMISPHERE","NETWORK","NETWORKKEY","GLASSERLABELNAME","RED","GREEN","BLUE","ALPHA"]]
    FCMat_df_info = FCMat_df_info.set_index("LABEL")
    FCMat_df.rename_axis(index=None, columns=None, inplace=True)
    FCMat_df_info.rename_axis(index=None, columns=None, inplace=True)
    return(FCMat_df, FCMat_df_info)
    # Return parcel-parcel matrix, and parcel informations

def Get_RP(GR_info, PP_info):
    print(LocalTime(), "Start getting RP network...")
    # Load parcel CIFTI file, include a total of 91282 grayordinates (59412 vertices and 31870 voxels)
    img_template = nib.load(args.parcelCIFTI)
    
    # Get grayordinate labels, these labels represent to which parcels
    # grayordinates belongs. Same as FCMat_df_info KEYVALUE column.
    print(LocalTime(), "Loading r-fMRI data...")
    grayordinate_label = img_template.get_fdata().squeeze().astype("int")
    
    # Get the brainmodel maps in CIFTI
    img_header = img_template.header
    img_grayord_maps = img_header.get_index_map(1)
    
    # Get the matrix that translates voxel IJK indicae to spatial XYZ coordinates
    img_ijk2xyz_matrix = img_grayord_maps[0].transformation_matrix_voxel_indices_ijk_to_xyz.matrix
    
    # Get the vertices/voxels start position in CX(Cortex), BS(Brain stem) and CB(Cerebellum)
    print(LocalTime(), "Get the grayordinates...")
    img_indexOfsets = [img_grayord_maps[1].index_offset, img_grayord_maps[2].index_offset, img_grayord_maps[7].index_offset, img_grayord_maps[10].index_offset]
    
    # Get the grayordinates
    cxl_vertex = {j:i for i,j in enumerate(img_grayord_maps[1].vertex_indices)}  # 29696  key is voxel label, values is index
    cxr_vertex = {j:i for i,j in enumerate(img_grayord_maps[2].vertex_indices)}  # 29716
    
    # range:left <= X <= right
    # 0-135, +59412
    # 136-450, +59687
    # 451-1178, +63806
    # 1179-1884, +83142
    # 1885-2648, +84560
    # 2649-2945, +86119
    # 2946-4005, +86676
    # 4006-5293, +88746
    cxl_voxel = np.array([i for i in img_grayord_maps[3].voxel_indices_ijk] + [i for i in img_grayord_maps[5].voxel_indices_ijk] + [i for i in img_grayord_maps[8].voxel_indices_ijk] + [i for i in img_grayord_maps[12].voxel_indices_ijk] + [i for i in img_grayord_maps[14].voxel_indices_ijk] + [i for i in img_grayord_maps[16].voxel_indices_ijk] + [i for i in img_grayord_maps[18].voxel_indices_ijk] + [i for i in img_grayord_maps[20].voxel_indices_ijk])  # 5293
    
    # range:left <= X <= right
    # 0-140, +59547
    # 141-472, +60002
    # 473-1227, +64534
    # 1228-1939, +83848
    # 1940-2734, +85324
    # 2735-2994, +86416
    # 2995-4004, +87736
    # 4005-5252, +90034
    cxr_voxel = np.array([i for i in img_grayord_maps[4].voxel_indices_ijk] + [i for i in img_grayord_maps[6].voxel_indices_ijk] + [i for i in img_grayord_maps[9].voxel_indices_ijk] + [i for i in img_grayord_maps[13].voxel_indices_ijk] + [i for i in img_grayord_maps[15].voxel_indices_ijk] + [i for i in img_grayord_maps[17].voxel_indices_ijk] + [i for i in img_grayord_maps[19].voxel_indices_ijk] + [i for i in img_grayord_maps[21].voxel_indices_ijk])  # 5252
    
    bs_voxel = np.array([i for i in img_grayord_maps[7].voxel_indices_ijk])  # 3472
    
    cb_voxel = np.array([i for i in img_grayord_maps[10].voxel_indices_ijk] + [i for i in img_grayord_maps[11].voxel_indices_ijk])  # 17853
    
    # Transform IJK to MNI
    def transform(matrix):
        matrix_ = np.insert(matrix, 3, np.ones(matrix.shape[0]), axis = 1)
        tmp = np.dot(img_ijk2xyz_matrix, matrix_.T).T
        tmp = np.delete(tmp, -1, axis=1)
        return tmp
    
    bs_voxel = transform(bs_voxel)
    cb_voxel = transform(cb_voxel)
    
    #region_info = pd.read_csv(GR_info, sep = "\t", index_col = "region_name")
    region_info = GR_info
    region_info.loc[(region_info["slab_type"] == "CX") & (region_info["structure_name"].str.contains("left")), "slab_type"] = "CX_L"
    region_info.loc[(region_info["slab_type"] == "CX") & (region_info["structure_name"].str.contains("right")), "slab_type"] = "CX_R"
    
    #parcel_info = pd.read_csv(PP_info, sep = "\t", index_col = "LABEL")
    parcel_info = PP_info
    
    # Create a regions * parcels matrix
    region_parcel = np.zeros((len(region_info), len(parcel_info)))
    region_parcel_df = pd.DataFrame(region_parcel, index = region_info.index.tolist(), columns = parcel_info.index.tolist())
    
    print(LocalTime(), "Matching brain regions and parcels")
    for i in region_info.itertuples():
        i_region_name = i[0]
        i_slab_type = i[4]
        i_coord = (i[11],i[12],i[13])
        
        # write MNI to a file for wb_command input
        with open(os.path.join(args.tmp, "tmp.txt"), "w") as f:
            for line in i_coord:
                f.write(str(line) + " ")
                
        if i_slab_type.startswith("CX_L"):
            os.system(args.wb + " -surface-closest-vertex" + " " + args.lsurf + " " + os.path.join(args.tmp, "tmp.txt") + " " + os.path.join(args.tmp, "output.txt"))
            with open(os.path.join(args.tmp, "output.txt"), "r") as f:
                vertex = [int(f.read())]  # return vertex number 1-32492
                
            if vertex[0] in cxl_vertex.keys():
                # True indicate this vertex is in CAB-NP network
                index = cxl_vertex[vertex[0]]  # this vertex correspond which number grayordinate in parcel CIFTI?
                grayordinate_index = grayordinate_label[index]
                grayordinate_parcel = parcel_info.loc[parcel_info["KEYVALUE"] == grayordinate_index,].index.tolist()[0]
                region_parcel_df.loc[i_region_name, grayordinate_parcel] = 1
            else:
                distances = np.sqrt(np.sum(np.square(cxl_voxel - i_coord), axis = 1))
                #print("CXL"+str(np.mean(distances)))
                if np.min(distances) <= 2:
                    index = np.squeeze(np.where(distances == np.min(distances)))
                    print(index)
                    index2 = []
                    for ii in index:
                        if 0 <= ii <= 135:
                            index2.append(ii+59412)
                        elif 136 <= ii <= 450:
                            index2.append(ii+59687)
                        elif 451 <= ii <= 1178:
                            index2.append(ii+63806)
                        elif 1179 <= ii <= 1884:
                            index2.append(ii+83142)
                        elif 1885 <= ii <= 2648:
                            index2.append(ii+84560)
                        elif 2649 <= ii <= 2945:
                            index2.append(ii+86119)
                        elif 2946 <= ii <= 4005:
                            index2.append(ii+86676)
                        elif 4006 <= ii <= 5293:
                            index2.append(ii+88746)
                            
                    grayordinate_index = grayordinate_label[index2]
                    if type(grayordinate_index) is np.ndarray:
                        for xx in grayordinate_index:
                            grayordinate_parcel = parcel_info.loc[parcel_info["KEYVALUE"] == xx,].index.tolist()[0]
                            region_parcel_df.loc[i_region_name, grayordinate_parcel] = 1
                    else:
                        grayordinate_index2 = grayordinate_index
                        grayordinate_parcel = parcel_info.loc[parcel_info["KEYVALUE"] == grayordinate_index2,].index.tolist()[0]
                        region_parcel_df.loc[i_region_name, grayordinate_parcel] = 1
                        
        elif i_slab_type.startswith("CX_R"):
            os.system(args.wb + " -surface-closest-vertex" + " " + args.rsurf + " " + os.path.join(args.tmp, "tmp.txt") + " " + os.path.join(args.tmp, "output.txt"))
            with open(os.path.join(args.tmp, "output.txt"), "r") as f:
                vertex = [int(f.read())]
                
            if vertex[0] in cxr_vertex.keys():
                index = cxr_vertex[vertex[0]] + img_indexOfsets[1]
                grayordinate_index = grayordinate_label[index]
                grayordinate_parcel = parcel_info.loc[parcel_info["KEYVALUE"] == grayordinate_index,].index.tolist()[0]
                region_parcel_df.loc[i_region_name, grayordinate_parcel] = 1
            else:
                distances = np.sqrt(np.sum(np.square(cxr_voxel - i_coord), axis = 1))
                #print("CXR"+str(np.mean(distances)))
                if np.min(distances) <= 2:
                    index = np.squeeze(np.where(distances == np.min(distances)))
                    index2 = []
                    for ii in index:
                        if 0 <= ii <= 140:
                            index2.append(ii+59547)
                        elif 141 <= ii <= 472:
                            index2.append(ii+60002)
                        elif 473 <= ii <= 1227:
                            index2.append(ii+64534)
                        elif 1228 <= ii <= 1939:
                            index2.append(ii+83848)
                        elif 1940 <= ii <= 2734:
                            index2.append(ii+85324)
                        elif 2735 <= ii <= 2994:
                            index2.append(ii+86416)
                        elif 2995 <= ii <= 4004:
                            index2.append(ii+87736)
                        elif 4005 <= ii <= 5252:
                            index2.append(ii+90034)
                            
                    grayordinate_index = grayordinate_label[index2]
                    if type(grayordinate_index) is np.ndarray:
                        for xx in grayordinate_index:
                            grayordinate_parcel = parcel_info.loc[parcel_info["KEYVALUE"] == xx,].index.tolist()[0]
                            region_parcel_df.loc[i_region_name, grayordinate_parcel] = 1
                    else:
                        grayordinate_index2 = grayordinate_index
                        grayordinate_parcel = parcel_info.loc[parcel_info["KEYVALUE"] == grayordinate_index2,].index.tolist()[0]
                        region_parcel_df.loc[i_region_name, grayordinate_parcel] = 1
                        
        elif i_slab_type == "BS":
            distances = np.sqrt(np.sum(np.square(bs_voxel - i_coord), axis = 1))
            if np.min(distances) <= 2:
                index = np.squeeze(np.where(distances == np.min(distances))) + img_indexOfsets[2]
                grayordinate_index = grayordinate_label[index]
                if type(grayordinate_index) is np.ndarray:
                    #grayordinate_index2 = grayordinate_index[0]
                    for xx in grayordinate_index:
                        grayordinate_parcel = parcel_info.loc[parcel_info["KEYVALUE"] == xx,].index.tolist()[0]
                        region_parcel_df.loc[i_region_name, grayordinate_parcel] = 1
                else:
                    grayordinate_index2 = grayordinate_index
                    grayordinate_parcel = parcel_info.loc[parcel_info["KEYVALUE"] == grayordinate_index2,].index.tolist()[0]
                    region_parcel_df.loc[i_region_name, grayordinate_parcel] = 1
                    
        elif i_slab_type == "CB":
            distances = np.sqrt(np.sum(np.square(cb_voxel - i_coord), axis = 1))
            if np.min(distances) <= 2:
                index = np.squeeze(np.where(distances == np.min(distances))) + img_indexOfsets[3]
                grayordinate_index = grayordinate_label[index]
                if type(grayordinate_index) is np.ndarray:
                    #grayordinate_index2 = grayordinate_index[0]
                    for xx in grayordinate_index:
                        grayordinate_parcel = parcel_info.loc[parcel_info["KEYVALUE"] == xx,].index.tolist()[0]
                        region_parcel_df.loc[i_region_name, grayordinate_parcel] = 1
                        
                else:
                    grayordinate_index2 = grayordinate_index
                    grayordinate_parcel = parcel_info.loc[parcel_info["KEYVALUE"] == grayordinate_index2,].index.tolist()[0]
                    region_parcel_df.loc[i_region_name, grayordinate_parcel] = 1
                    #region_parcel_df = region_parcel_df.rename_axis("REGION", axis = 0)
                    
    return(region_parcel_df)
    # Return region*parcel

def Get_RR(PP, RP):
    print(LocalTime(), "Start getting RR network...")
    FCMat_parcel = PP
    region_parcel = RP
    print(LocalTime(), "Calculating...")
    FCMat_region = np.dot(np.dot(region_parcel, FCMat_parcel), region_parcel.T)
    #FCMat_region = np.abs(np.tanh(FCMat_region))
    FCMat_region = FCMat_region - np.diag(np.diag(FCMat_region))
    FCMat_region[FCMat_region == 0] = 1e-8
    print(LocalTime(), "Get region-region connectivity...")
    FCMat_region_df = pd.DataFrame(FCMat_region, index = region_parcel.index.tolist(), columns = region_parcel.index.tolist())
    #FCMat_region_df = FCMat_region_df.rename_axis("REGION", axis = 0)
    return(FCMat_region_df)
    # Return region-region matrix

def Get_GT():
    # Import GRNdb network
    print(LocalTime(), "Start getting GT network...")
    print(LocalTime(), "Loading GRNdb data...")
    grndb_df = pd.read_csv(args.GRNdb, sep = "\t")
    
    # Drop nan in GRNdb newtwork
    grndb_df = grndb_df.loc[pd.notna(grndb_df["Genie3Weight"])]
    grndb_df = grndb_df.reset_index(drop = True)
    
    grndb_gene = grndb_df["gene"].drop_duplicates().tolist()
    grndb_tf = grndb_df["TF"].drop_duplicates().tolist()
    print(LocalTime(), "Get", len(grndb_gene), "unique genes...")
    print(LocalTime(), "Get", len(grndb_tf), "transcription factors...")
    print(LocalTime(), "Get", len(grndb_df), "links...")
    
    gene_tf = pd.DataFrame(np.zeros((len(grndb_gene), len(grndb_tf))), columns = grndb_tf, index = grndb_gene)
    for ind,row in grndb_df.iterrows():
        #print(ind)
        gene_tf.loc[row["gene"], row["TF"]] = row["Genie3Weight"]
        
    return(gene_tf)
    # Return GRN

def Get_Common_Genes_And_TF():
    print("AAA")
    Allen = pd.read_csv(args.AllenProbes).dropna(axis=0, how='any').reset_index(drop = True)
    GRNdb = pd.read_csv(args.GRNdb, sep = "\t")
    GRNdb = GRNdb.loc[pd.notna(GRNdb["Genie3Weight"])]
    GRNdb = GRNdb.reset_index(drop = True)
    GRNdb_symbol = pd.Series(GRNdb["TF"].tolist() + GRNdb["gene"].tolist()).drop_duplicates().reset_index(drop=True)
    #GRNdb_symbol = pd.Series(GRNdb["gene"].tolist()).drop_duplicates().reset_index(drop=True)
    GRNdb_gene = GRNdb["gene"].copy().drop_duplicates().reset_index(drop=True)
    GRNdb_TF = GRNdb["TF"].copy().drop_duplicates().reset_index(drop=True)
    
    GTEx = pd.read_csv(args.GTEx, sep="\t", skiprows=2)
    GTEx = GTEx.loc[:,["Name", "Description"]]
    
    gene_history = pd.read_csv(args.genehistory, sep="\t")
    gene_history = gene_history.loc[gene_history["#tax_id"] == 9606,].reset_index(drop=True)
    gene_history["GeneID"] = gene_history["GeneID"].astype("str")
    gene_history["Discontinued_GeneID"] = gene_history["Discontinued_GeneID"].astype("str")
    
    # Update Allen entrez ID to the latest 
    Allen["gene_id"] = Allen["gene_id"].astype("str")
    Allen["entrez_id"] = Allen["entrez_id"].astype("int").astype("str")
    Allen["New_entrez"] = Allen["entrez_id"]
    for i in range(len(Allen)):
        new_id = Allen.loc[i, "New_entrez"]
        while new_id in gene_history["Discontinued_GeneID"].values:
            #print(i,new_id)
            new_id = gene_history[gene_history["Discontinued_GeneID"] == new_id]["GeneID"].tolist()[0]
        Allen.loc[i,"New_entrez"] = new_id
        
    # Match GRNdb and GTEx to get GRNdb ensemble ID
    gene2ensembl = pd.read_csv(args.gene2ensembl, sep = "\t")
    gene2ensembl = gene2ensembl.loc[gene2ensembl["#tax_id"] == 9606,].reset_index(drop=True)
    gene2ensembl["GeneID"] = gene2ensembl["GeneID"].astype("str")
    gene2ensembl = gene2ensembl[gene2ensembl["GeneID"].isin(Allen["New_entrez"].tolist())].reset_index(drop=True)
    print("BBB")
    GTEx["Name"] = GTEx["Name"].apply(lambda x:x[0:15])
    GTEx = GTEx[GTEx["Description"].isin(GRNdb_symbol.tolist())].reset_index(drop=True)
    
    GRNdb = GTEx.copy()
    GRNdb.columns = ["GTEx_Ensembl", "GRNdb_Symbol"]
    
    # Match Allen, GRNdb, GTEx
    final_dt = pd.DataFrame({"Allen_Entrez":Allen["entrez_id"], "New_Entrez":Allen["New_entrez"], "Allen_Symbol":Allen["gene_symbol"], "Allen_Probe":Allen["probe_name"]})
    final_dt["Ensg_ByAllen_OldID"] = np.nan
    final_dt["Ensg_ByAllen_NewID"] = np.nan
    final_dt["GRNdb_ByEnsg_Old"] = np.nan
    final_dt["GRNdb_ByEnsg_New"] = np.nan
    
    for i in range(len(final_dt)):
        this_allen_entrez = final_dt.loc[i, "Allen_Entrez"]
        this_df = gene2ensembl.loc[gene2ensembl["GeneID"] == this_allen_entrez]
        if len(this_df) == 0:
            final_dt.loc[i, "Ensg_ByAllen_OldID"] = '---'
        else:
            final_dt.loc[i, "Ensg_ByAllen_OldID"] = "|".join(list(set(this_df["Ensembl_gene_identifier"].tolist())))
            
    for i in range(len(final_dt)):
        new_allen_entrez = final_dt.loc[i, "New_Entrez"]
        this_df = gene2ensembl.loc[gene2ensembl["GeneID"] == new_allen_entrez]
        if len(this_df) == 0:
            final_dt.loc[i, "Ensg_ByAllen_NewID"] = "---"
        else:
            final_dt.loc[i, "Ensg_ByAllen_NewID"] = "|".join(list(set(this_df["Ensembl_gene_identifier"].tolist())))
            
    for i in range(len(final_dt)):
        ensg_by_old = final_dt.loc[i, "Ensg_ByAllen_OldID"]
        this_df = GRNdb.loc[GRNdb["GTEx_Ensembl"] == ensg_by_old]
        if len(this_df) == 0:
            final_dt.loc[i, "GRNdb_ByEnsg_Old"] = "---"
        else:
            final_dt.loc[i, "GRNdb_ByEnsg_Old"] = "|".join(list(set(this_df["GRNdb_Symbol"].tolist())))
            
    for i in range(len(final_dt)):
        ensg_by_new = final_dt.loc[i, "Ensg_ByAllen_NewID"]
        this_df = GRNdb.loc[GRNdb["GTEx_Ensembl"] == ensg_by_new]
        if len(this_df) == 0:
            final_dt.loc[i, "GRNdb_ByEnsg_New"] = "---"
        else:
            final_dt.loc[i, "GRNdb_ByEnsg_New"] = "|".join(list(set(this_df["GRNdb_Symbol"].tolist())))
            
    unique_allen_entrez = [i for n, i in enumerate(final_dt["Allen_Entrez"].tolist()) if i not in final_dt["Allen_Entrez"].tolist()[:n]]
    
    final_keep = pd.DataFrame({"Allen_Raw_ID":unique_allen_entrez, "New_Entrez":np.nan, "Allen_Symbol":np.nan, "Allen_Probe":np.nan, "Ensembl":np.nan, "GRNdb":np.nan})
    
    for i in range(len(final_keep)):
        this_e = final_keep.loc[i, "Allen_Raw_ID"]
        this_df = final_dt.loc[final_dt["Allen_Entrez"] == this_e]
        
        final_keep.loc[i, "New_Entrez"] = "|".join(list(set(this_df["New_Entrez"].tolist())))
        final_keep.loc[i, "Allen_Symbol"] = "|".join(list(set(this_df["Allen_Symbol"].tolist())))
        final_keep.loc[i, "Allen_Probe"] = "|".join(list(set(this_df["Allen_Probe"].tolist())))
        
        e1 = this_df["Ensg_ByAllen_OldID"].tolist()[0].split("|")
        e2 = this_df["Ensg_ByAllen_NewID"].tolist()[0].split("|")
        e_all = np.array(list(set(e1 + e2)))
        if all(e_all == "---"):
            final_keep.loc[i, "Ensembl"] = "---"
        else:
            final_keep.loc[i, "Ensembl"] = "|".join(e_all[e_all != "---"])
            
        g1 = this_df["GRNdb_ByEnsg_Old"].tolist()[0].split("|")
        g2 = this_df["GRNdb_ByEnsg_New"].tolist()[0].split("|")
        g_all = np.array(list(set(g1 + g2)))
        if all(g_all == "---"):
            final_keep.loc[i, "GRNdb"] = "---"
        else:
            final_keep.loc[i, "GRNdb"] = "|".join(g_all[g_all != "---"])
            
    ok_df = final_keep.loc[final_keep["GRNdb"] != "---"].reset_index(drop = True)
    # This is for debug
    #ok_df.to_csv("ok_df.csv")
    
    def ReadFasta(fa_file):
        fa_dict = {}
        for line in open(fa_file):
            if line[0] == ">":
                key = line.split()[0][1:]
                fa_dict[key] = ""
            else:
                fa_dict[key] += (line.strip())
        return(fa_dict)
        
    rna1 = ReadFasta(args.HumanRNA1)
    rna2 = ReadFasta(args.HumanRNA2)
    rna = {**rna1, **rna2}
    rna_name = rna.keys()
    
    pep1 = ReadFasta(args.HumanPEP1)
    pep2 = ReadFasta(args.HumanPEP2)
    pep = {**pep1, **pep2}
    pep_name = pep.keys()
    
    ok_ensembl = gene2ensembl[gene2ensembl["Ensembl_gene_identifier"].isin(ok_df["Ensembl"])].reset_index(drop=True)
    ok_ensembl["RNA_nucleotide_accession.version.LEN"] = np.nan
    ok_ensembl["protein_accession.version.LEN"] = np.nan
    
    for i in range(len(ok_ensembl)):
        rna_id = ok_ensembl.loc[i, "RNA_nucleotide_accession.version"]
        pep_id = ok_ensembl.loc[i, "protein_accession.version"]
        
        if rna_id != "-" and rna_id in rna_name:
            ok_ensembl.loc[i, "RNA_nucleotide_accession.version.LEN"] = int(len(rna[rna_id]))
            
        if pep_id != "-" and pep_id in pep_name:
            ok_ensembl.loc[i, "protein_accession.version.LEN"] = int(len(pep[pep_id]))
            
    unique_ok_ensembl_geneID = [i for n, i in enumerate(ok_ensembl["GeneID"].tolist()) if i not in ok_ensembl["GeneID"].tolist()[:n]]
    
    ok_ensembl_unique = pd.DataFrame({"GeneID":unique_ok_ensembl_geneID, "Ensembl_gene_identifier":np.nan, "RNA_nucleotide_accession.version":np.nan, "Ensembl_rna_identifier":np.nan, "protein_accession.version":np.nan, "Ensembl_protein_identifier":np.nan, "RNA_nucleotide_accession.version.LEN":np.nan, "protein_accession.version.LEN":np.nan})
    
    for i in range(len(ok_ensembl_unique)):
        this_id = ok_ensembl_unique.loc[i, "GeneID"]
        this_df = ok_ensembl.loc[ok_ensembl["GeneID"] == this_id].reset_index(drop=True)
        this_df_clean = this_df.dropna(axis=0, how="any").reset_index(drop=True)
        if len(this_df_clean) == 0:
            ok_ensembl_unique.loc[i, "Ensembl_gene_identifier"] = this_df.loc[0, "Ensembl_gene_identifier"]
            ok_ensembl_unique.loc[i, "RNA_nucleotide_accession.version"] = this_df.loc[0, "RNA_nucleotide_accession.version"]
            ok_ensembl_unique.loc[i, "Ensembl_rna_identifier"] = this_df.loc[0, "Ensembl_rna_identifier"]
            ok_ensembl_unique.loc[i, "protein_accession.version"] = this_df.loc[0, "protein_accession.version"]
            ok_ensembl_unique.loc[i, "Ensembl_protein_identifier"] = this_df.loc[0, "Ensembl_protein_identifier"]
            ok_ensembl_unique.loc[i, "RNA_nucleotide_accession.version.LEN"] = this_df.loc[0, "RNA_nucleotide_accession.version.LEN"]
            ok_ensembl_unique.loc[i, "protein_accession.version.LEN"] = this_df.loc[0, "protein_accession.version.LEN"]
        else:
            ok_ensembl_unique.loc[i, "Ensembl_gene_identifier"] = this_df_clean.loc[0, "Ensembl_gene_identifier"]
            ok_ensembl_unique.loc[i, "RNA_nucleotide_accession.version"] = this_df_clean.loc[0, "RNA_nucleotide_accession.version"]
            ok_ensembl_unique.loc[i, "Ensembl_rna_identifier"] = this_df_clean.loc[0, "Ensembl_rna_identifier"]
            ok_ensembl_unique.loc[i, "protein_accession.version"] = this_df_clean.loc[0, "protein_accession.version"]
            ok_ensembl_unique.loc[i, "Ensembl_protein_identifier"] = this_df_clean.loc[0, "Ensembl_protein_identifier"]
            ok_ensembl_unique.loc[i, "RNA_nucleotide_accession.version.LEN"] = this_df_clean.loc[0, "RNA_nucleotide_accession.version.LEN"]
            ok_ensembl_unique.loc[i, "protein_accession.version.LEN"] = this_df_clean.loc[0, "protein_accession.version.LEN"]
            
    ok_ensembl_unique = ok_ensembl_unique.dropna(axis=0, how="any").reset_index(drop=True)
    ok_ensembl_unique = ok_ensembl_unique.rename(columns={"GeneID":"Allen_Raw_ID"})
    ok_ensembl_unique.set_index("Allen_Raw_ID", inplace=True)
    
    ok_final = ok_df.join(ok_ensembl_unique, on ="Allen_Raw_ID", how="inner").reset_index(drop=True)
    ok_final.drop("Ensembl_gene_identifier", axis=1, inplace=True)
    ok_final.columns =["Allen_Entrez", "Current_Entrez", "Allen_Symbol", "Allen_Probe", "Ensembl_GeneID", "GRNdb_Symbol", "GeneBank_GeneID", "Ensembl_RnaID", "GeneBank_ProteinID", "Ensembl_ProteinID", "Gene_Length", "Protein_Length"]
    
    ok_gene = ok_final[ok_final["GRNdb_Symbol"].isin(GRNdb_gene)].reset_index(drop=True)
    ok_tf = ok_final[ok_final["GRNdb_Symbol"].isin(GRNdb_TF)].reset_index(drop=True)
    
    ok_gene["GeneID"] = "E"+np.array(ok_gene["Allen_Entrez"])
    ok_tf["TFID"] = "T"+np.array(ok_tf["Allen_Entrez"])
    
    ok_gene["Allen_Entrez"] = ok_gene["Allen_Entrez"].astype("int")
    ok_gene = ok_gene.sort_values(by="Allen_Entrez").reset_index(drop=True)
    ok_gene["Allen_Entrez"] = ok_gene["Allen_Entrez"].astype("str")
    
    ok_tf["Allen_Entrez"] = ok_tf["Allen_Entrez"].astype("int")
    ok_tf = ok_tf.sort_values(by="Allen_Entrez").reset_index(drop=True)
    ok_tf["Allen_Entrez"] = ok_tf["Allen_Entrez"].astype("str")
    
    ok_gene_rna = {}
    ok_gene_pep = {}
    for i in range(len(ok_gene)):
        this_rna = ok_gene.loc[i, "GeneBank_GeneID"]
        this_pep = ok_gene.loc[i, "GeneBank_ProteinID"]
        this_entrez = ok_gene.loc[i, "Allen_Entrez"]
        ok_gene_rna["E"+this_entrez] = rna[this_rna]
        ok_gene_pep["E"+this_entrez] = pep[this_pep]
        
    ok_tf_rna = {}
    ok_tf_pep = {}
    for i in range(len(ok_tf)):
        this_rna = ok_tf.loc[i, "GeneBank_GeneID"]
        this_pep = ok_tf.loc[i, "GeneBank_ProteinID"]
        this_entrez = ok_tf.loc[i, "Allen_Entrez"]
        ok_tf_rna["T"+this_entrez] = rna[this_rna]
        ok_tf_pep["T"+this_entrez] = pep[this_pep]
        
    return(ok_gene, ok_gene_rna, ok_gene_pep, ok_tf, ok_tf_rna, ok_tf_pep)
    # Return match ID, RNA seq, Protein seq

def Get_TT(TF_PEP):
    # make a blast tmp dir and write pep.fasta
    os.system("mkdir "+args.tmp+"/BLAST")
    fa_name = list(TF_PEP.keys())
    fa_seq = list(TF_PEP.values())
    fa = open(args.tmp+"/BLAST/TF_SEQ.fa", "w")
    for i in range(len(TF_PEP)):
        fa.writelines(">"+fa_name[i]+"\n")
        fa.writelines(fa_seq[i]+"\n")
    fa.close()
    
    # makeblastdb & blastp
    os.system(args.makeblastdb+" -in "+args.tmp+"/BLAST/TF_SEQ.fa -dbtype prot")
    os.system(args.blastp+" -query "+args.tmp+"/BLAST/TF_SEQ.fa -db "+args.tmp+"/BLAST/TF_SEQ.fa -word_size 2 -evalue 10 -num_threads 4 -out "+args.tmp+"/BLAST/TT_similarity.csv -outfmt 10")        
    
    tt_df = pd.read_csv(args.tmp+"/BLAST/TT_similarity.csv", header = None)
    tt_df.columns = ["Query","Subject","Identity","Alignment_length","Mismatches","Gap_opens","Q_start","Q_end","S_start","S_end","E_value","Bit_score"]
    tt_df = tt_df[tt_df["Query"]!=tt_df["Subject"]]
    tt_df = tt_df[~np.array(tt_df.loc[:,["Query","Subject"]].duplicated())].reset_index(drop=True)
    
    tt_name = tt_df["Query"].drop_duplicates().tolist()
    for i in tt_name:
        this_tt_df = tt_df.loc[tt_df["Query"] == i,]
        threshold = np.percentile(this_tt_df["Identity"], 85)
        this_tt_remove = this_tt_df.loc[this_tt_df["Identity"] < threshold].index.tolist()
        tt_df.drop(this_tt_remove, inplace=True)
        
    tt_df = tt_df.reset_index(drop=True)
    
    tt_final = pd.DataFrame(0, index=tt_name, columns=tt_name)
    for i in range(len(tt_df)):
        tt_final.loc[tt_df.loc[i, "Query"], tt_df.loc[i, "Subject"]] = tt_df.loc[i, "Identity"]
        tt_final.loc[tt_df.loc[i, "Subject"], tt_df.loc[i, "Query"]] = tt_df.loc[i, "Identity"]
        
    for i in tt_name:
        tt_final.loc[i, i] = 100
        
    return(tt_final)
    # Return TT

def Get_GG(GT, TT):
    gg_df = np.dot(np.dot(GT.values, TT.values), GT.values.T)*1e7
    gg_df = pd.DataFrame(gg_df, index=GT.index.tolist(), columns=GT.index.tolist())
    return(gg_df)

def Enhance_GG(GG):
    edf = pd.read_csv(args.enhanceMT, sep="\t", header=0, index_col=0)
    eGG = np.multiply(GG, edf)
    return(eGG)


def Get_DD():
    vd = pd.read_csv(args.VariantDisease, sep="\t", header=0)
    vd = vd[vd["diseaseType"] == "disease"].reset_index(drop=True)
    dis_snp_list = vd.groupby("diseaseId").agg({"snpId":set}).reset_index().to_dict(orient="records")
    dis_snp_dict = {}
    for i in dis_snp_list:
        dis_snp_dict[i["diseaseId"]] = i["snpId"]
        
    dis_jcd = {}
    for i in dis_snp_dict.keys():
        this_list = []
        for j in dis_snp_dict.keys():
            this_list.append(len(dis_snp_dict[i] & dis_snp_dict[j]) / len(dis_snp_dict[i] | dis_snp_dict[j]))
        dis_jcd[i] = this_list  
        
    dis_dis = pd.DataFrame(dis_jcd)
    dis_dis.index = dis_dis.columns.tolist()
    dis_dis = dis_dis * 100
    return(dis_dis)

def Get_GD(G_G, D_D):
    dis = pd.read_csv(args.GeneDisease, header=0, sep="\t")
    dis["geneId"] = "E"+dis["geneId"].apply(str)
    dis = dis[dis["diseaseType"] == "disease"].reset_index(drop=True)
    keep_disease = set(dis["diseaseId"]) & set(D_D.index)
    dis = dis[dis["diseaseId"].isin(keep_disease)].reset_index(drop=True)
    
    dis_gene = dis[dis["geneId"].isin(G_G.index.tolist())].reset_index(drop = True)
    
    final_disease = dis_gene["diseaseId"].drop_duplicates().tolist()
    final_gene = dis_gene["geneId"].drop_duplicates().tolist()
    
    dis_gene_list = dis_gene.groupby("diseaseId").agg({"geneId":set}).reset_index().to_dict(orient="records")
    dis_gene_dict = {}
    for i in dis_gene_list:
        dis_gene_dict[i["diseaseId"]] = i["geneId"]
        
    gene_dis = pd.DataFrame(0, index=final_gene, columns=final_disease)
    
    for i in dis_gene_dict.keys():
        #print(i)
        related_genes = dis_gene_dict[i]
        gene_dis.loc[related_genes, i] = 1
        
    return(gene_dis)

def Get_GTEx(Common_genes):
    gtex_tpm = pd.read_csv(args.GTEx, sep="\t", skiprows=2)
    gtex_sample = pd.read_csv(args.GTExSample, sep="\t")
    brain_sample = gtex_sample[(gtex_sample["SMTS"] == "Brain") & (gtex_sample["SMAFRZE"] == "RNASEQ")].reset_index(drop=True)
    brain_tpm = gtex_tpm.loc[:,["Name","Description"]+brain_sample["SAMPID"].tolist()]
    brain_tpm["Name"] = brain_tpm["Name"].str[0:15]
    brain_tpm.set_index("Name", inplace=True, drop=False)
    common_tpm = brain_tpm.loc[Common_genes["Ensembl_GeneID"]]
    common_tpm["Description"] = Common_genes["GeneID"].tolist()
    common_tpm.drop("Name", axis=1, inplace=True)
    common_tpm.set_index("Description", drop=True, inplace=True)
    common_tpm.index.name = None
    return(common_tpm)

def StackAutoEncoder(df, encoding_dim):
    x = df.values
    x_train, x_test, y_train, y_test = train_test_split(x, x, test_size=0.2)
    x_train = x_train.astype('float32') / 1
    x_test = x_test.astype('float32') / 1
    input_img = Input(shape=(len(x[0]),))
    
    encoded = Dense(512, activation='relu')(input_img)
    encoded = Dense(256, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)
    
    decoded = Dense(256, activation='relu')(encoder_output)
    decoded = Dense(512, activation='relu')(decoded)
    decoded = Dense(len(x[0]), activation='tanh')(decoded)
    
    autoencoder = Model(inputs=input_img, outputs=decoded)
    encoder = Model(inputs=input_img, outputs=encoder_output)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    x = x.astype('float32')
    autoencoder.fit(x, x, epochs=5, batch_size=128, shuffle=True, validation_data=(x_test, x_test))
    
    encoded_imgs = encoder.predict(x)
    NewEncoded_imgs = []
    counter = 0
    while counter < len(encoded_imgs):
        Row = []
        Row.extend(encoded_imgs[counter])
        NewEncoded_imgs.append(Row)
        counter = counter + 1
    gene_feature = pd.DataFrame(NewEncoded_imgs)
    gene_feature.index = df.index.tolist()
    return(gene_feature)
 
def Get_All_Matrix():
    Common_genes, Common_genes_RNA, Common_genes_PEP, TF, TF_RNA, TF_PEP = Get_Common_Genes_And_TF()
    Common_genes.to_csv("./Process_data/GeneID.txt", sep="\t")
    G_info_RNA = Common_genes_RNA
    G_info_PEP = Common_genes_PEP
    T_info_RNA = TF_RNA
    T_info_PEP = TF_PEP
    GRNdb_GeneID_dict = dict(zip(Common_genes["GRNdb_Symbol"].tolist(), Common_genes["GeneID"].tolist()))
    GRNdb_TFID_dict = dict(zip(TF["GRNdb_Symbol"].tolist(), TF["TFID"].tolist()))
    # G_R and R_G
    G_R, R_info = Get_GR()
    G_R = G_R.loc[Common_genes["GeneID"],]
    R_G = G_R.T
    
    # P_P
    P_P, P_info = Get_PP()
   
    # R_P and P_R
    R_P = Get_RP(R_info, P_info)
    P_R = R_P.T
    
    # R_R
    R_R = Get_RR(P_P, R_P)
    
    # G_T and T_G
    G_T = Get_GT()
    G_T.rename(index=GRNdb_GeneID_dict, inplace=True)
    G_T = G_T.loc[Common_genes["GeneID"],]
    G_T.rename(columns=GRNdb_TFID_dict, inplace=True)
    G_T = G_T.loc[:,TF["TFID"]]
    T_G = G_T.T
    
    # T_T
    T_T = Get_TT(TF_PEP)
    T_T = T_T.loc[TF["TFID"]]
    T_T = T_T.loc[:,TF["TFID"]]
    
    # G_G
    G_G = Get_GG(G_T, T_T)
    
    # T_R and R_T
    T_R = pd.DataFrame(np.dot(T_G, G_R), index=T_T.index.tolist(), columns=R_R.index.tolist())
    R_T = T_R.T
    
    # G_P and P_G
    G_P = pd.DataFrame(np.dot(G_R, R_P), index=G_R.index.tolist(), columns=P_P.index.tolist())
    P_G = G_P.T
    
    # D_D
    D_D = Get_DD()
    
    # G_D and D_G
    G_D = Get_GD(G_G, D_D)
    D_G = G_D.T
    
    # Final update gene and disease
    G_D_count = G_D.apply(sum, axis=1)
    G_D_count = G_D_count[G_D_count > 0]
    final_gene = pd.Series(G_D_count.index.tolist())
    final_disease = pd.Series(G_D.columns.tolist())
    
    
    G_G = G_G.loc[final_gene,final_gene]
    T_T = T_T
    R_R = R_R
    P_P = P_P
    D_D = D_D.loc[final_disease, final_disease]
    G_D = G_D
    D_G = D_G
    G_T = G_T.loc[final_gene,]
    T_G = G_T.T
    G_R = G_R.loc[final_gene,]
    R_G = G_R.T
    T_R = T_R
    R_T = R_T
    G_P = G_P.loc[final_gene,]
    P_G = G_P.T
    G_R = G_R.loc[final_gene,]   
    R_G = G_R.T
    P_R = P_R
    R_P = P_R.T
    
    G_GTEx = Get_GTEx(Common_genes)
    G_GTEx = G_GTEx.loc[G_G.index.tolist()]
    
    G_feature = StackAutoEncoder(G_GTEx, 64)
    G_feature.to_csv("./Process_data/Gene_features.txt", sep="\t")
    
    D_feature = StackAutoEncoder(D_D, 64)
    D_feature.to_csv("./Process_data/Disease_features.txt", sep="\t")
    
    gtr_adj = np.vstack([np.hstack([G_G,G_T,G_R]), np.hstack([T_G,T_T,T_R]), np.hstack([R_G,R_T,R_R])])    
    gpr_adj=np.vstack([np.hstack([G_G,G_P,G_R]), np.hstack([P_G,P_P,P_R]), np.hstack([R_G,R_P,R_R])])
    
    os.system("rm -r ./Process_data/BLAST ./Process_data/tmp.txt ./Process_data/output.txt")
    
    gtr_adj_index = G_G.index.tolist()+T_G.index.tolist()+R_G.index.tolist()
    gpr_adj_index = G_G.index.tolist()+P_G.index.tolist()+R_G.index.tolist()
    
    gtr_adj_df = pd.DataFrame(gtr_adj, index=gtr_adj_index, columns=gtr_adj_index)
    gpr_adj_df = pd.DataFrame(gpr_adj, index=gpr_adj_index, columns=gpr_adj_index)
    
    gtr_adj_df.to_csv("./Process_data/GTR.txt", sep="\t")
    gpr_adj_df.to_csv("./Process_data/GPR.txt", sep="\t")
    
    
    
    G_G.to_csv("./Process_data/G_G.txt", sep="\t")
    T_T.to_csv("./Process_data/T_T.txt", sep="\t")
    R_R.to_csv("./Process_data/R_R.txt", sep="\t")
    P_P.to_csv("./Process_data/P_P.txt", sep="\t")
    D_D.to_csv("./Process_data/D_D.txt", sep="\t")
    G_D.to_csv("./Process_data/G_D.txt", sep="\t")
    D_G.to_csv("./Process_data/D_G.txt", sep="\t")
    G_T.to_csv("./Process_data/G_T.txt", sep="\t")
    T_G.to_csv("./Process_data/T_G.txt", sep="\t")
    G_R.to_csv("./Process_data/G_R.txt", sep="\t")
    R_G.to_csv("./Process_data/R_G.txt", sep="\t")
    T_R.to_csv("./Process_data/T_R.txt", sep="\t")
    R_T.to_csv("./Process_data/R_T.txt", sep="\t")
    G_P.to_csv("./Process_data/G_P.txt", sep="\t")
    P_G.to_csv("./Process_data/P_G.txt", sep="\t")
    G_R.to_csv("./Process_data/G_R.txt", sep="\t")
    R_G.to_csv("./Process_data/R_G.txt", sep="\t")
    P_R.to_csv("./Process_data/P_R.txt", sep="\t")
    R_P.to_csv("./Process_data/R_P.txt", sep="\t")
    G_GTEx.to_csv("./Process_data/G_GTEx.txt", sep="\t")
    return(gtr_adj_df, gpr_adj_df, G_G, T_T, R_R, P_P, D_D, G_D, D_G, G_T, T_G, G_R, R_G, T_R, R_T, G_P, P_G, G_R, R_G, P_R, R_P, G_GTEx)

if __name__ == "__main__":
    GTR, GPR, G_G, T_T, R_R, P_P, D_D, G_D, D_G, G_T, T_G, G_R, R_G, T_R, R_T, G_P, P_G, G_R, R_G, P_R, R_P, G_GTEx = Get_All_Matrix()
    
    
    

