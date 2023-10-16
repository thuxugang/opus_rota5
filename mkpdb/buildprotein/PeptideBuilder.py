import tensorflow as tf
import numpy as np
from mkpdb.myclass import Atoms

def get_norm(v):
    return tf.norm(v, axis=-1)
        
def get_angle(p1, p2, p3):
    
    v1 = p1 - p2
    v2 = p3 - p2

    v1_norm = get_norm(v1)
    v2_norm = get_norm(v2)
    c = tf.reduce_sum(v1*v2, -1)/(v1_norm * v2_norm)

    c = tf.clip_by_value(c, -0.999999, 0.999999)
    
    return tf.math.acos(c)/np.pi*180

def get_angle2(v1, v2):
    
    v1_norm = get_norm(v1)
    v2_norm = get_norm(v2)
    c = tf.reduce_sum(v1*v2, -1)/(v1_norm * v2_norm)

    c = tf.clip_by_value(c, -0.999999, 0.999999)
    
    return tf.math.acos(c)/np.pi*180

def get_dihedral(p1, p2, p3, p4):
    
    c1 = p1 - p2
    c2 = p2 - p3
    c3 = p3 - p4
    
    v1 = tf.linalg.cross(c2, c1)
    v2 = tf.linalg.cross(c3, c2)
    v3 = tf.linalg.cross(v2, v1)
    
    return tf.sign(tf.reduce_sum(v3*c2,-1))*get_angle2(v1,v2)
        
def calculateCoordinates(c1, c2, c3, L, ang, di):

    d2 = tf.stack([L*tf.math.cos(ang/180*np.pi),
                   L*tf.math.cos(di/180*np.pi)*tf.math.sin(ang/180*np.pi),
                   L*tf.math.sin(di/180*np.pi)*tf.math.sin(ang/180*np.pi)])
    ab = c2 - c1
    bc = c3 - c2
    bc = bc/get_norm(bc)
    n = tf.linalg.cross(ab, bc)
    n = n/get_norm(n)
    ab = tf.linalg.cross(n, bc)
    
    mtr = tf.stack([-bc, ab, n])
    mtr = tf.transpose(mtr)

    bc = tf.experimental.numpy.dot(mtr, d2)
    cc = c3 + bc
    
    return tf.cast(cc, tf.float32)
    
def makeSer(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer):
    '''Creates a Serine residue'''
    CB_OG_length=geo.CB_OG_length
    CA_CB_OG_angle=geo.CA_CB_OG_angle
    
    OG = calculateCoordinates(N, CA, CB, CB_OG_length, CA_CB_OG_angle, rotamer[0])

    return [OG], ["OG"]

def makeCys(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer):
    '''Creates a Cysteine residue'''
    
    CB_SG_length= geo.CB_SG_length
    CA_CB_SG_angle= geo.CA_CB_SG_angle

    SG = calculateCoordinates(N, CA, CB, CB_SG_length, CA_CB_SG_angle, rotamer[0])

    return [SG], ["SG"]

def makeVal(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer):
    '''Creates a Valine residue'''
    CB_CG1_length=geo.CB_CG1_length
    CA_CB_CG1_angle=geo.CA_CB_CG1_angle
    
    CB_CG2_length=geo.CB_CG2_length
    CG1_CB_CG2_angle=geo.CG1_CB_CG2_angle
    CG1_CA_CB_CG2_diangle=geo.CG1_CA_CB_CG2_diangle
        
    CG1 = calculateCoordinates(N, CA, CB, CB_CG1_length, CA_CB_CG1_angle, rotamer[0])
    CG2 = calculateCoordinates(CG1, CA, CB, CB_CG2_length, CG1_CB_CG2_angle, CG1_CA_CB_CG2_diangle)

    return [CG1, CG2], ["CG1", "CG2"]

def makeIle(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer):
    '''Creates an Isoleucine residue'''
    
    CB_CG1_length=geo.CB_CG1_length
    CA_CB_CG1_angle=geo.CA_CB_CG1_angle
    
    CB_CG2_length=geo.CB_CG2_length
    CG1_CB_CG2_angle=geo.CG1_CB_CG2_angle
    CG1_CA_CB_CG2_diangle=geo.CG1_CA_CB_CG2_diangle

    CG1_CD1_length= geo.CG1_CD1_length
    CB_CG1_CD1_angle= geo.CB_CG1_CD1_angle
        
    CG1 = calculateCoordinates(N, CA, CB, CB_CG1_length, CA_CB_CG1_angle, rotamer[0])
    CG2 = calculateCoordinates(CG1, CA, CB, CB_CG2_length, CG1_CB_CG2_angle, CG1_CA_CB_CG2_diangle)
    CD1 = calculateCoordinates(CA, CB, CG1, CG1_CD1_length, CB_CG1_CD1_angle, rotamer[1])

    return [CG1, CG2, CD1], ["CG1", "CG2", "CD1"]

def makeLeu(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer):
    '''Creates a Leucine residue'''

    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle= geo.CA_CB_CG_angle
    
    CG_CD1_length=geo.CG_CD1_length
    CB_CG_CD1_angle=geo.CB_CG_CD1_angle

    CG_CD2_length=geo.CG_CD2_length
    CD1_CG_CD2_angle=geo.CD1_CG_CD2_angle
    CD1_CB_CG_CD2_diangle=geo.CD1_CB_CG_CD2_diangle

    CG = calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, rotamer[0])
    CD1 = calculateCoordinates(CA, CB, CG, CG_CD1_length, CB_CG_CD1_angle, rotamer[1])
    CD2 = calculateCoordinates(CD1, CB, CG, CG_CD2_length, CD1_CG_CD2_angle, CD1_CB_CG_CD2_diangle)

    return [CG, CD1, CD2], ["CG", "CD1", "CD2"]
    
def makeThr(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer):
    '''Creates a Threonine residue'''
    
    CB_OG1_length=geo.CB_OG1_length
    CA_CB_OG1_angle=geo.CA_CB_OG1_angle
        
    CB_CG2_length=geo.CB_CG2_length
    OG1_CB_CG2_angle=geo.OG1_CB_CG2_angle
    OG1_CA_CB_CG2_diangle= geo.OG1_CA_CB_CG2_diangle

    OG1 = calculateCoordinates(N, CA, CB, CB_OG1_length, CA_CB_OG1_angle, rotamer[0])
    CG2 = calculateCoordinates(OG1, CA, CB, CB_CG2_length, OG1_CB_CG2_angle, OG1_CA_CB_CG2_diangle)

    return [OG1, CG2], ["OG1", "CG2"]

def makeArg(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer):
    '''Creates an Arginie residue'''

    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle= geo.CA_CB_CG_angle
    
    CG_CD_length=geo.CG_CD_length
    CB_CG_CD_angle=geo.CB_CG_CD_angle
    
    CD_NE_length=geo.CD_NE_length
    CG_CD_NE_angle=geo.CG_CD_NE_angle
    
    NE_CZ_length=geo.NE_CZ_length
    CD_NE_CZ_angle=geo.CD_NE_CZ_angle

    CZ_NH1_length=geo.CZ_NH1_length
    NE_CZ_NH1_angle=geo.NE_CZ_NH1_angle
    CD_NE_CZ_NH1_diangle=geo.CD_NE_CZ_NH1_diangle

    NH1_NH2_length=geo.NH1_NH2_length
    NE_CZ_NH2_angle=geo.NE_CZ_NH2_angle
    NH1_NE_CZ_NH2_diangle=geo.NH1_NE_CZ_NH2_diangle
        
    CG = calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, rotamer[0])
    CD = calculateCoordinates(CA, CB, CG, CG_CD_length, CB_CG_CD_angle, rotamer[1])
    NE = calculateCoordinates(CB, CG, CD, CD_NE_length, CG_CD_NE_angle, rotamer[2])
    CZ = calculateCoordinates(CG, CD, NE, NE_CZ_length, CD_NE_CZ_angle, rotamer[3])
    NH1 = calculateCoordinates(CD, NE, CZ, CZ_NH1_length, NE_CZ_NH1_angle, CD_NE_CZ_NH1_diangle)
    NH2 = calculateCoordinates(NH1, NE, CZ, NH1_NH2_length, NE_CZ_NH2_angle, NH1_NE_CZ_NH2_diangle)

    return [CG, CD, NE, CZ, NH1, NH2], ["CG", "CD", "NE", "CZ", "NH1", "NH2"]

def makeLys(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer):
    '''Creates a Lysine residue'''

    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle

    CG_CD_length=geo.CG_CD_length
    CB_CG_CD_angle=geo.CB_CG_CD_angle

    CD_CE_length=geo.CD_CE_length
    CG_CD_CE_angle=geo.CG_CD_CE_angle

    CE_NZ_length=geo.CE_NZ_length
    CD_CE_NZ_angle=geo.CD_CE_NZ_angle
    
    CG = calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, rotamer[0])
    CD = calculateCoordinates(CA, CB, CG, CG_CD_length, CB_CG_CD_angle, rotamer[1])
    CE = calculateCoordinates(CB, CG, CD, CD_CE_length, CG_CD_CE_angle, rotamer[2])
    NZ = calculateCoordinates(CG, CD, CE, CE_NZ_length, CD_CE_NZ_angle, rotamer[3])

    return [CG, CD, CE, NZ], ["CG", "CD", "CE", "NZ"]

def makeAsp(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer):
    '''Creates an Aspartic Acid residue'''

    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle

    CG_OD1_length=geo.CG_OD1_length
    CB_CG_OD1_angle=geo.CB_CG_OD1_angle

    CG_OD2_length=geo.CG_OD2_length
    CB_CG_OD2_angle=geo.CB_CG_OD2_angle
    OD1_CB_CG_OD2_diangle=geo.OD1_CB_CG_OD2_diangle

    CG = calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, rotamer[0])
    OD1 = calculateCoordinates(CA, CB, CG, CG_OD1_length, CB_CG_OD1_angle, rotamer[1])
    OD2 = calculateCoordinates(OD1, CB, CG, CG_OD2_length, CB_CG_OD2_angle, OD1_CB_CG_OD2_diangle)

    return [CG, OD1, OD2], ["CG", "OD1", "OD2"]

def makeAsn(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer):
    '''Creates an Asparagine residue'''
    
    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle
    
    CG_OD1_length=geo.CG_OD1_length
    CB_CG_OD1_angle=geo.CB_CG_OD1_angle
    
    CG_ND2_length=geo.CG_ND2_length
    CB_CG_ND2_angle=geo.CB_CG_ND2_angle
    OD1_CB_CG_ND2_diangle=geo.OD1_CB_CG_ND2_diangle
    
    CG = calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, rotamer[0])
    OD1 = calculateCoordinates(CA, CB, CG, CG_OD1_length, CB_CG_OD1_angle, rotamer[1])
    ND2 = calculateCoordinates(OD1, CB, CG, CG_ND2_length, CB_CG_ND2_angle, OD1_CB_CG_ND2_diangle)

    return [CG, OD1, ND2], ["CG", "OD1", "ND2"]

def makeGlu(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer):
    '''Creates a Glutamic Acid residue'''
    
    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle

    CG_CD_length=geo.CG_CD_length
    CB_CG_CD_angle=geo.CB_CG_CD_angle

    CD_OE1_length=geo.CD_OE1_length
    CG_CD_OE1_angle=geo.CG_CD_OE1_angle

    CD_OE2_length=geo.CD_OE2_length
    CG_CD_OE2_angle=geo.CG_CD_OE2_angle
    OE1_CG_CD_OE2_diangle=geo.OE1_CG_CD_OE2_diangle
    
    CG = calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, rotamer[0])
    CD = calculateCoordinates(CA, CB, CG, CG_CD_length, CB_CG_CD_angle, rotamer[1])
    OE1 = calculateCoordinates(CB, CG, CD, CD_OE1_length, CG_CD_OE1_angle, rotamer[2])
    OE2 = calculateCoordinates(OE1, CG, CD, CD_OE2_length, CG_CD_OE2_angle, OE1_CG_CD_OE2_diangle)

    return [CG, CD, OE1, OE2], ["CG", "CD", "OE1", "OE2"]

def makeGln(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer):
    '''Creates a Glutamine residue'''
    
    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle

    CG_CD_length=geo.CG_CD_length
    CB_CG_CD_angle=geo.CB_CG_CD_angle
    
    CD_OE1_length=geo.CD_OE1_length
    CG_CD_OE1_angle=geo.CG_CD_OE1_angle
    
    CD_NE2_length=geo.CD_NE2_length
    CG_CD_NE2_angle=geo.CG_CD_NE2_angle
    OE1_CG_CD_NE2_diangle=geo.OE1_CG_CD_NE2_diangle
    
    CG = calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, rotamer[0])
    CD = calculateCoordinates(CA, CB, CG, CG_CD_length, CB_CG_CD_angle, rotamer[1])
    OE1 = calculateCoordinates(CB, CG, CD, CD_OE1_length, CG_CD_OE1_angle, rotamer[2])
    NE2 = calculateCoordinates(OE1, CG, CD, CD_NE2_length, CG_CD_NE2_angle, OE1_CG_CD_NE2_diangle)

    return [CG, CD, OE1, NE2], ["CG", "CD", "OE1", "NE2"]

def makeMet(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer):
    '''Creates a Methionine residue'''

    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle
    
    CG_SD_length=geo.CG_SD_length
    CB_CG_SD_angle=geo.CB_CG_SD_angle
    
    SD_CE_length=geo.SD_CE_length
    CG_SD_CE_angle=geo.CG_SD_CE_angle
    
    CG = calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, rotamer[0])
    SD = calculateCoordinates(CA, CB, CG, CG_SD_length, CB_CG_SD_angle, rotamer[1])
    CE = calculateCoordinates(CB, CG, SD, SD_CE_length, CG_SD_CE_angle, rotamer[2])

    return [CG, SD, CE], ["CG", "SD", "CE"]

def makeHis(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer):
    '''Creates a Histidine residue'''

    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle
    
    CG_ND1_length=geo.CG_ND1_length
    CB_CG_ND1_angle=geo.CB_CG_ND1_angle
    
    CG_CD2_length=geo.CG_CD2_length
    CB_CG_CD2_angle=geo.CB_CG_CD2_angle
    ND1_CB_CG_CD2_diangle=geo.ND1_CB_CG_CD2_diangle

    ND1_CE1_length=geo.ND1_CE1_length
    CG_ND1_CE1_angle=geo.CG_ND1_CE1_angle
    CB_CG_ND1_CE1_diangle=geo.CB_CG_ND1_CE1_diangle
    
    CD2_NE2_length=geo.CD2_NE2_length
    CG_CD2_NE2_angle=geo.CG_CD2_NE2_angle
    CB_CG_CD2_NE2_diangle=geo.CB_CG_CD2_NE2_diangle
    
    CG = calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, rotamer[0])
    ND1 = calculateCoordinates(CA, CB, CG, CG_ND1_length, CB_CG_ND1_angle, rotamer[1])
    CD2 = calculateCoordinates(ND1, CB, CG, CG_CD2_length, CB_CG_CD2_angle, ND1_CB_CG_CD2_diangle)
    CE1 = calculateCoordinates(CB, CG, ND1, ND1_CE1_length, CG_ND1_CE1_angle, CB_CG_ND1_CE1_diangle)
    NE2 = calculateCoordinates(CB, CG, CD2, CD2_NE2_length, CG_CD2_NE2_angle, CB_CG_CD2_NE2_diangle)

    return [CG, ND1, CD2, CE1, NE2], ["CG", "ND1", "CD2", "CE1", "NE2"]

def makePro(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer):
    '''Creates a Proline residue'''
    
    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle
    
    CG_CD_length=geo.CG_CD_length
    CB_CG_CD_angle=geo.CB_CG_CD_angle
    
    CG = calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, rotamer[0])
    CD = calculateCoordinates(CA, CB, CG, CG_CD_length, CB_CG_CD_angle, rotamer[1])

    return [CG, CD], ["CG", "CD"]

def makePhe(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer):
    '''Creates a Phenylalanine residue'''
    
    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle

    CG_CD1_length=geo.CG_CD1_length
    CB_CG_CD1_angle=geo.CB_CG_CD1_angle

    CG_CD2_length=geo.CG_CD2_length
    CB_CG_CD2_angle=geo.CB_CG_CD2_angle
    CD1_CB_CG_CD2_diangle=geo.CD1_CB_CG_CD2_diangle
    
    CD1_CE1_length=geo.CD1_CE1_length
    CG_CD1_CE1_angle=geo.CG_CD1_CE1_angle
    CB_CG_CD1_CE1_diangle=geo.CB_CG_CD1_CE1_diangle

    CD2_CE2_length=geo.CD2_CE2_length
    CG_CD2_CE2_angle=geo.CG_CD2_CE2_angle
    CB_CG_CD2_CE2_diangle=geo.CB_CG_CD2_CE2_diangle

    CE1_CZ_length=geo.CE1_CZ_length
    CD1_CE1_CZ_angle=geo.CD1_CE1_CZ_angle
    CG_CD1_CE1_CZ_diangle=geo.CG_CD1_CE1_CZ_diangle
        
    CG = calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, rotamer[0])
    CD1 = calculateCoordinates(CA, CB, CG, CG_CD1_length, CB_CG_CD1_angle, rotamer[1])
    CD2 = calculateCoordinates(CD1, CB, CG, CG_CD2_length, CB_CG_CD2_angle, CD1_CB_CG_CD2_diangle)
    CE1 = calculateCoordinates(CB, CG, CD1, CD1_CE1_length, CG_CD1_CE1_angle, CB_CG_CD1_CE1_diangle)
    CE2 = calculateCoordinates(CB, CG, CD2, CD2_CE2_length, CG_CD2_CE2_angle, CB_CG_CD2_CE2_diangle)
    CZ = calculateCoordinates(CG, CD1, CE1, CE1_CZ_length, CD1_CE1_CZ_angle, CG_CD1_CE1_CZ_diangle)

    return [CG, CD1, CD2, CE1, CE2, CZ], ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]

def makeTyr(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer):
    '''Creates a Tyrosine residue'''
    
    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle

    CG_CD1_length=geo.CG_CD1_length
    CB_CG_CD1_angle=geo.CB_CG_CD1_angle
    
    CG_CD2_length=geo.CG_CD2_length
    CB_CG_CD2_angle=geo.CB_CG_CD2_angle
    CD1_CB_CG_CD2_diangle=geo.CD1_CB_CG_CD2_diangle
    
    CD1_CE1_length=geo.CD1_CE1_length
    CG_CD1_CE1_angle=geo.CG_CD1_CE1_angle
    CB_CG_CD1_CE1_diangle=geo.CB_CG_CD1_CE1_diangle

    CD2_CE2_length=geo.CD2_CE2_length
    CG_CD2_CE2_angle=geo.CG_CD2_CE2_angle
    CB_CG_CD2_CE2_diangle=geo.CB_CG_CD2_CE2_diangle

    CE1_CZ_length=geo.CE1_CZ_length
    CD1_CE1_CZ_angle=geo.CD1_CE1_CZ_angle
    CG_CD1_CE1_CZ_diangle=geo.CG_CD1_CE1_CZ_diangle

    CZ_OH_length=geo.CZ_OH_length
    CE1_CZ_OH_angle=geo.CE1_CZ_OH_angle
    CD1_CE1_CZ_OH_diangle=geo.CD1_CE1_CZ_OH_diangle
        
    CG = calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, rotamer[0])
    CD1 = calculateCoordinates(CA, CB, CG, CG_CD1_length, CB_CG_CD1_angle, rotamer[1])
    CD2 = calculateCoordinates(CD1, CB, CG, CG_CD2_length, CB_CG_CD2_angle, CD1_CB_CG_CD2_diangle)
    CE1 = calculateCoordinates(CB, CG, CD1, CD1_CE1_length, CG_CD1_CE1_angle, CB_CG_CD1_CE1_diangle)
    CE2 = calculateCoordinates(CB, CG, CD2, CD2_CE2_length, CG_CD2_CE2_angle, CB_CG_CD2_CE2_diangle)
    CZ = calculateCoordinates(CG, CD1, CE1, CE1_CZ_length, CD1_CE1_CZ_angle, CG_CD1_CE1_CZ_diangle)
    OH = calculateCoordinates(CD1, CE1, CZ, CZ_OH_length, CE1_CZ_OH_angle, CD1_CE1_CZ_OH_diangle)

    return [CG, CD1, CD2, CE1, CE2, CZ, OH], ["CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"]

def makeTrp(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer):
    '''Creates a Tryptophan residue'''

    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle

    CG_CD1_length=geo.CG_CD1_length
    CB_CG_CD1_angle=geo.CB_CG_CD1_angle

    CG_CD2_length=geo.CG_CD2_length
    CB_CG_CD2_angle=geo.CB_CG_CD2_angle
    CD1_CB_CG_CD2_diangle=geo.CD1_CB_CG_CD2_diangle
    
    CD1_NE1_length=geo.CD1_NE1_length
    CG_CD1_NE1_angle=geo.CG_CD1_NE1_angle
    CB_CG_CD1_NE1_diangle=geo.CB_CG_CD1_NE1_diangle

    CD2_CE2_length=geo.CD2_CE2_length
    CG_CD2_CE2_angle=geo.CG_CD2_CE2_angle
    CB_CG_CD2_CE2_diangle=geo.CB_CG_CD2_CE2_diangle

    CD2_CE3_length=geo.CD2_CE3_length
    CG_CD2_CE3_angle=geo.CG_CD2_CE3_angle
    CB_CG_CD2_CE3_diangle=geo.CB_CG_CD2_CE3_diangle

    CE2_CZ2_length=geo.CE2_CZ2_length
    CD2_CE2_CZ2_angle=geo.CD2_CE2_CZ2_angle
    CG_CD2_CE2_CZ2_diangle=geo.CG_CD2_CE2_CZ2_diangle

    CE3_CZ3_length=geo.CE3_CZ3_length
    CD2_CE3_CZ3_angle=geo.CD2_CE3_CZ3_angle
    CG_CD2_CE3_CZ3_diangle=geo.CG_CD2_CE3_CZ3_diangle

    CZ2_CH2_length=geo.CZ2_CH2_length
    CE2_CZ2_CH2_angle=geo.CE2_CZ2_CH2_angle
    CD2_CE2_CZ2_CH2_diangle=geo.CD2_CE2_CZ2_CH2_diangle
        
    CG = calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, rotamer[0])
    CD1 = calculateCoordinates(CA, CB, CG, CG_CD1_length, CB_CG_CD1_angle, rotamer[1])
    CD2 = calculateCoordinates(CD1, CB, CG, CG_CD2_length, CB_CG_CD2_angle, CD1_CB_CG_CD2_diangle)
    NE1 = calculateCoordinates(CB, CG, CD1, CD1_NE1_length, CG_CD1_NE1_angle, CB_CG_CD1_NE1_diangle)
    CE2 = calculateCoordinates(CB, CG, CD2, CD2_CE2_length, CG_CD2_CE2_angle, CB_CG_CD2_CE2_diangle)
    CE3 = calculateCoordinates(CB, CG, CD2, CD2_CE3_length, CG_CD2_CE3_angle, CB_CG_CD2_CE3_diangle)
    CZ2 = calculateCoordinates(CG, CD2, CE2, CE2_CZ2_length, CD2_CE2_CZ2_angle, CG_CD2_CE2_CZ2_diangle)
    CZ3 = calculateCoordinates(CG, CD2, CE3, CE3_CZ3_length, CD2_CE3_CZ3_angle, CG_CD2_CE3_CZ3_diangle)
    CH2 = calculateCoordinates(CD2, CE2, CZ2, CZ2_CH2_length, CE2_CZ2_CH2_angle, CD2_CE2_CZ2_CH2_diangle)

    return [CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2], ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"]

def get_coordinate(rotamer, residue, geo, atoms_matrix):
    
    resname = residue.resname
    
    all_atoms = []
    all_atoms_names = []
    
    N = atoms_matrix[residue.main_chain_atoms_matrixid["N"]]
    CA = atoms_matrix[residue.main_chain_atoms_matrixid["CA"]]
    C = atoms_matrix[residue.main_chain_atoms_matrixid["C"]]
    O = atoms_matrix[residue.main_chain_atoms_matrixid["O"]]
    
    all_atoms.append(N)
    all_atoms.append(CA)
    all_atoms.append(C)
    all_atoms.append(O)

    all_atoms_names.extend(["N", "CA", "C", "O", "CB"])
    
    if resname == "G":
        CB = residue.atoms["CB"].position
        all_atoms.append(CB)
        return all_atoms, all_atoms_names
    
    CB = atoms_matrix[residue.main_chain_atoms_matrixid["CB"]]
    all_atoms.append(CB)
    
    atoms = []
    all_atoms_name = []
    if(resname=='A'):
        pass
    elif(resname=='S'):
        atoms, all_atoms_name = makeSer(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer)
    elif(resname=='C'):
        atoms, all_atoms_name = makeCys(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer)
    elif(resname=='V'):
        atoms, all_atoms_name = makeVal(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer)
    elif(resname=='I'):
        atoms, all_atoms_name = makeIle(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer)
    elif(resname=='L'):
        atoms, all_atoms_name = makeLeu(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer)
    elif(resname=='T'):
        atoms, all_atoms_name = makeThr(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer)
    elif(resname=='R'):
        atoms, all_atoms_name = makeArg(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer)
    elif(resname=='K'):
        atoms, all_atoms_name = makeLys(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer)
    elif(resname=='D'):
        atoms, all_atoms_name = makeAsp(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer)
    elif(resname=='E'):
        atoms, all_atoms_name = makeGlu(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer)
    elif(resname=='N'):
        atoms, all_atoms_name = makeAsn(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer)
    elif(resname=='Q'):
        atoms, all_atoms_name = makeGln(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer)
    elif(resname=='M'):
        atoms, all_atoms_name = makeMet(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer)
    elif(resname=='H'):
        atoms, all_atoms_name = makeHis(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer)
    elif(resname=='P'):
        atoms, all_atoms_name = makePro(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer)
    elif(resname=='F'):
        atoms, all_atoms_name = makePhe(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer)
    elif(resname=='Y'):
        atoms, all_atoms_name = makeTyr(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer)
    elif(resname=='W'):
        atoms, all_atoms_name = makeTrp(N, CA, C, O, CB, geo, atoms_matrix, residue, rotamer)
    else:
        print("PeptideBuilder.initialize_res wrong")
    
    all_atoms.extend(atoms)
    all_atoms_names.extend(all_atoms_name)
    return all_atoms, all_atoms_names

def get_cb(geo, residue):
    
    segID = residue.resid
    resname = residue.resname
    
    N = residue.atoms['N'].position
    CA = residue.atoms['CA'].position
    C = residue.atoms['C'].position
    
    carbon_b = calculateCoordinates(C, N, CA, geo.CA_CB_length, geo.C_CA_CB_angle, geo.N_C_CA_CB_diangle)
    CB = Atoms.Atom(-1, "CB", resname, segID, carbon_b)
    
    return CB
