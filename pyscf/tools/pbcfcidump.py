#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Based on fcidump.py

'''
FCIDUMP functions (write, read) for complex Hamiltonian for periodic systems

'#P' indicates that these lines exist for parallelism.

Based on fcidump.py, but with additional symmetry specification for k-point momentum
as used by HANDE.

Currently just RHF and needs modified kccsd_rhf to store all eris.

It also calculates modified exchange integrals using the truncated exchange
treatment which differ from the exchange using the bare Coulomb interaction.
These can be written in the same format as an FCIDUMP file, but without a header.

'''
from functools import reduce
import copy
import numpy
import pyscf.pbc
import datetime, uuid, h5py
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.cc.kccsd_rhf import KRCCSD
try:  # P
    from mpi4py import MPI  # P
    mpi4py_avail = True  # P
except ImportError:  # P
    mpi4py_avail = False  # P

# [todo] - allow read in from __config__, too.
DEFAULT_FLOAT_FORMAT = ' (%.16g,%.16g)'
TOL = 1e-10
# [todo] - allow orbsyms?
MOLPRO_ORBSYM = False


def write_head(fout, nmo, nelec, ms, nprop, propbitlen, orbsym=None):
    '''Write header of FCIDUMP file.

    Args:
        fout : file
            FCIDUMP file.
        nmo : int
            Number of (molecular) orbitals.
        nelec : int or List of ints [todo]
            Number of electrons.
        ms : int
            Overall spin.
        nprop : List of int
            Number of k points in each dimension.  [todo] - check
        propbitlen : int
            Number of bits in an element of orbsym corresponding to
            a dimension of k.  It means that an element in orbsym
            (which is an integer) can represent k (which is a list of
            integers).  If k is three dimensional, each third of the
            bit representation of an element of orbsym belongs to an
            an element of k.  The lengths of this third, in number of
            bits, is propbitlen.
            [todo] - improve? HANDE has an example (by Charlie)
    Kwargs:
        orbsym : List of ints, optional
            Integers labelling symmetry of the orbitals.  If not
            supplied, will assign label '1' to all orbitals, assigning
            them the same symmetry.
    '''
    if not isinstance(nelec, (int, numpy.number)):
        ms = abs(nelec[0] - nelec[1])
        nelec = nelec[0] + nelec[1]
    fout.write(' &FCI NORB=%4d,NELEC=%2d,MS2=%d,\n' % (nmo, nelec, ms))
    if orbsym is not None and len(orbsym) > 0:
        fout.write('  ORBSYM=%s\n' % ','.join([str(x) for x in orbsym]))
    else:
        fout.write('  ORBSYM=%s\n' % ('1,' * (nmo-1) + '1'))
    fout.write('  NPROP=%4d %4d %4d\n' % tuple(nprop))
    fout.write('  PROPBITLEN=%4d\n' % propbitlen)
    fout.write('  ISYM=1,\n')
    fout.write(' &END\n')

def get_eri(eri, kp, kq, kr, ks, i, j, k, l, no):
    '''Get electron repulsion integrals (ERIs) for a given set of orbitals.

    Args:
        eri : numpy array
            Contains ERIs divided by number of k points, with indices
            [ka,kc,kb,a,c,b,d].
        kp, kq, kr, ks : int 
            k-point indices of each orbital. 
        i, j, k, l : int
            Global orbital indices. 
        no : int 
            Number of occupied orbitals. 
    '''
    if i < no and j < no and k < no and l < no:
        # oooo
        v = eri.oooo[kp, kr, kq, i, k, j, l]
    elif i < no and j < no and k < no and l >= no:
        # ooov
        v = eri.ooov[kp, kr, kq, i, k, j, l-no]
    elif i < no and j < no and k >= no and l < no:
        # iojokvlo => ovoo
        v = eri.ooov[kq, ks, kp, j, l, i, k-no].conj()
    elif i < no and j >= no and k < no and l < no:
        # iojvkolo => oovo
        v = eri.ooov[kr, kp, ks, k, i, l, j-no]
    elif i >= no and j < no and k < no and l < no:
        # vooo
        v = eri.ooov[ks, kq, kr, l, j, k, i-no].conj()
    elif i < no and j < no and k >= no and l >= no:
        # iojokvlv => ovov
        v = eri.ovov[kp, kr, kq, i, k-no, j, l-no]
    elif i < no and j >= no and k < no and l >= no:
        # iojvkolv => oovv
        v = eri.oovv[kp, kr, kq, i, k, j-no, l-no]
    elif i >= no and j < no and k < no and l >= no:
        # voov
        v = eri.voov[kp, kr, kq, i-no, k, j, l-no]
    elif i < no and j >= no and k >= no and l < no:
        # ovvo
        v = eri.voov[kr, kp, ks, k-no, i, l, j-no]
    elif i >= no and j < no and k >= no and l < no:
        # ivjokvlo => vvoo
        v = eri.oovv[kq, ks, kp, j, l, i-no, k-no].conj()
    elif i >= no and j >= no and k < no and l < no:
        # ivjvkolo => vovo
        v = eri.ovov[kr, kp, ks, k, i-no, l, j-no]
    elif i >= no and j >= no and k >= no and l < no:
        # vvvo
        v = eri.vovv[kq, ks, kp, j-no, l, i-no, k-no].conj()
    elif i >= no and j >= no and k < no and l >= no:
        # ivjvkolv => vovv
        v = eri.vovv[kp, kr, kq, i-no, k, j-no, l-no]
    elif i >= no and j < no and k >= no and l >= no:
        # ivjokvlv => vvov
        v = eri.vovv[ks, kq, kr, l-no, j, k-no, i-no].conj()
    elif i < no and j >= no and k >= no and l >= no:
        # ovvv
        v = eri.vovv[kr, kp, ks, k-no, i, l-no, j-no]
    elif i >= no and j >= no and k >= no and l >= no:
        # vvvv
        v = eri.vvvv[kp, kr, kq, i-no, k-no, j-no, l-no]
    else:
        raise RuntimeError()
    return v

def tri_ind(i, j):
    ''' Calculate index of element (i, j) in upper triangule of symmetric matrix.

    Args: 
        i : int 
            Row index.
        j : int 
            Column index.
    '''
    return i * (i - 1) // 2 + j 

def tri_ind_reorder(i, j):
    ''' Calculate index of element (i, j) in upper triangule of symmetric matrix with reordering of indices.
    Args: 
        i : int 
            Row index.
        j : int 
            Column index.
    '''
    if i >= j:
        return tri_ind(i, j)
    else:
        return tri_ind(j, i)

def get_hande_index_coulomb(i, j, k, l, kp, kq, kr, ks, nor, mapping):
    ''' Calculate HANDE storage index of a Coulomb integral. 

    Args:
        kp, kq, kr, ks : int 
            k-point indices of each orbital. 
        i, j, k, l : int
            Global orbital indices. 
        nor : int 
            Number of orbitals.
        mapping: dict
            Maps the PySCF orbital indices to those seen by HANDE.
    '''
    ii = mapping[kp*nor + i] + 1
    aa = mapping[kq*nor + j] + 1
    jj = mapping[kr*nor + k] + 1
    bb = mapping[ks*nor + l] + 1
    orbs = (ii, jj, aa, bb)
    maxv = max(orbs)
    maxv_n = sum(x == maxv for x in (ii, jj, aa, bb))
    conj = False 
    eswap = False
    if maxv_n == 1:
        conj = (maxv in (aa, bb))
        eswap = (maxv in (jj, bb))
    elif maxv_n > 1:
        if ii == jj and ii == maxv:
            conj = False 
            eswap = bb > aa 
        elif aa == bb and aa == maxv:
            conj = True 
            eswap = jj > ii 
        elif ii == aa and ii == maxv:
            conj = bb > jj
            eswap = False 
        elif ii == bb and ii == maxv:
            conj = jj > aa 
            eswap = conj 
        elif jj == aa and jj == maxv:
            conj = ii > bb
            eswap = not conj 
        elif jj == bb and jj == maxv: 
            conj = ii < aa 
            eswap = True 
    if conj:
        ii, aa = aa, ii 
        jj, bb = bb, jj 
    if eswap:
        ii, jj = jj, ii 
        aa, bb = bb, aa
    ia = tri_ind(ii, aa)
    jb = tri_ind_reorder(jj, bb)
    # Index as spin orbitals with zero-based index 
    if jj < bb:
        index = 2 * tri_ind(ia, jb) - 1 
    else:
        index = 2 * tri_ind(ia, jb) - 2
    return index

def get_hande_index_exchange(i, j, k, l, kp, kq, kr, ks, nor, mapping):
    ''' Calculate HANDE storage index of an exchange integral. 

    Args:
        kp, kq, kr, ks : int 
            k-point indices of each orbital. 
        i, j, k, l : int
            Global orbital indices. 
        nor : int 
            Number of orbitals.
        mapping: dict
            Maps the PySCF orbital indices to those seen by HANDE.
    '''
    ii = mapping[kp*nor + i] + 1
    aa = mapping[kq*nor + j] + 1
    jj = mapping[kr*nor + k] + 1
    bb = mapping[ks*nor + l] + 1
    orbs = (ii, jj, aa, bb)
    maxv = max(orbs)
    maxv_n = sum(x == maxv for x in (ii, jj, aa, bb))
    conj = False 
    eswap = False
    if ii == bb:
        if jj == aa:
            if ii < jj:
                eswap = True 
        else:
            if jj < aa:
                conj = True 
                eswap = True
    elif jj == aa:
        if ii < bb:
            conj = True 
        else:
            eswap = True 

    if conj:
        ii, aa = aa, ii 
        jj, bb = bb, jj 
    if eswap:
        ii, jj = jj, ii 
        aa, bb = bb, aa

    index_tri_ind = tri_ind(jj, aa) - 1 
    index_repeat = ii - 1
    return index_tri_ind, index_repeat

def write_eri(fout, eri, kconserv, tol=TOL,
              float_format=DEFAULT_FLOAT_FORMAT):
    '''Write electron repulsion integrals (ERIs) to FCIDUMP.

    Args:
        fout : file
            FCIDUMP file.
        eri : numpy array
            Contains ERIs divided by number of k points, with indices
            [ka,kc,kb,a,c,b,d].
        kconserv : function
            Pass in three k point indices and get fourth one where
            overall momentum is conserved.

    Kwargs:
        tol : float, optional
            Below this value integral is not written to file.
            The default is TOL.
        float_format : str, optional
            Format of integral for writing.
            The default is DEFAULT_FLOAT_FORMAT.
    '''
    output_format = float_format + ' %4d %4d %4d %4d\n'
    nkpts = eri.oooo.shape[0]
    no = eri.oooo.shape[-1]
    nv = eri.vvvv.shape[-1]
    nor = no+nv
    for kp in range(nkpts):
        for kq in range(nkpts):
            for kr in range(nkpts):
                ks = kconserv[kp, kq, kr]
                # The documentation in pyscf/pbc/lib/kpts_helper.py in
                # get_kconserv is inconsistent with the actual code (and
                # physics). [k*(1) l(1) | m*(2) n(2)] = <km|ln> is the
                # integral. kconserve gives n given klm, such that
                # l-k=n-m (not k-l=n-m)
                for i in range(nor):
                    for j in range(nor):
                        for k in range(nor):
                            for l in range(nor):
                                # Stored as [ka,kc,kb,a,c,b,d] <- (ab|cd)
                                v = get_eri(eri, kp, kq, kr, ks, i, j, k, l, no)
                                if abs(v) > tol:
                                    fout.write(output_format % (
                                        v.real, v.imag, nor*kp+i+1, nor*kq+j+1,
                                        nor*kr+k+1, nor*ks+l+1))

def write_eri_HDF5(group_integrals, eri, kconserv, mapping, tol=TOL):
    '''Write electron repulsion integrals (ERIs) to system .H5.
    
    Args:
        group_integrals : h5py.Group 
            HDF5 group which stores the integrals.
        eri : numpy array
            Contains ERIs divided by number of k points, with indices
            [ka,kc,kb,a,c,b,d].
        kconserv : function
            Pass in three k point indices and get fourth one where
            overall momentum is conserved.
        mapping: dict
            Maps the PySCF orbital indices to those seen by HANDE.

    Kwargs:
        tol : float, optional
            Below this value integral is not written to file.
            The default is TOL.
    '''
    nkpts = eri.oooo.shape[0]
    no = eri.oooo.shape[-1]
    nv = eri.vvvv.shape[-1]
    nor = no+nv
    nspat = len(mapping)
    nuniq = nspat * (nspat + 1) // 2
    npair = nuniq * (nuniq + 1) // 2
    ntotal  = 2 * npair
    coulomb_ints_real, coulomb_ints_imag = numpy.zeros((ntotal), dtype=numpy.float64), numpy.zeros((ntotal), dtype=numpy.float64)
    for kp in range(nkpts):
        for kq in range(nkpts):
            for kr in range(nkpts):
                ks = kconserv[kp, kq, kr]
                # The documentation in pyscf/pbc/lib/kpts_helper.py in
                # get_kconserv is inconsistent with the actual code (and
                # physics). [k*(1) l(1) | m*(2) n(2)] = <km|ln> is the
                # integral. kconserve gives n given klm, such that
                # l-k=n-m (not k-l=n-m)
                for i in range(nor):
                    for j in range(nor):
                        for k in range(nor):
                            for l in range(nor):
                                # Stored as [ka,kc,kb,a,c,b,d] <- (ab|cd)
                                v = get_eri(eri, kp, kq, kr, ks, i, j, k, l, no)
                                index = get_hande_index_coulomb(i, j, k, l, kp, kq, kr, ks, nor, mapping)
                                if abs(v) > tol:
                                    coulomb_ints_real[index] = v.real 
                                    coulomb_ints_imag[index] = v.imag
    group_integrals.create_dataset('coulomb_ints_im_ispin01', data=coulomb_ints_imag)  
    group_integrals.create_dataset('coulomb_ints_ispin01', data=coulomb_ints_real)    


def write_exchange_integrals(fout, xints, ki, nkpts, nor, tol=TOL,
                             float_format=DEFAULT_FLOAT_FORMAT):
    '''Write extra exchange electron repulsion integrals to FCIDUMP.

    Args:
        fout : file
            FCIDUMP file.
        xints : numpy array, dim: (MO at k point, all MO, all MO)
            extra exchange integrals <pi|iq>_x
        ki : int
            k index for orb i in <pi|iq>_x
        nkpts : int
            Number of k points
        nor : int
            Number of orbitals at a k point.
        mapping: dict
            Maps the PySCF orbital indices to those seen by HANDE.

    Kwargs:
        tol : float, optional
            Below this value integral is not written to file.
            The default is TOL.
        float_format : str, optional
            Format of integral for writing.
            The default is DEFAULT_FLOAT_FORMAT.
    '''
    output_format = float_format + ' %4d %4d %4d %4d\n'
    for kj in range(nkpts):
        for kk in range(nkpts):
            for i in range(nor):
                for j in range(nor):
                    for k in range(nor):
                        v = xints[i, kj*nor+j, kk*nor+k]/nkpts
                        if abs(v) > tol:
                            fout.write(output_format % (
                                v.real, v.imag, nor*ki+i+1, nor*kj+j+1,
                                nor*kk+k+1, nor*ki+i+1))

def write_exchange_integrals_HDF5(group_integrals, xints, ki, nkpts, nor, mapping, tol=TOL):
    '''Write extra exchange electron repulsion integrals to system .H5.

    Args:
        group_integrals : h5py.Group 
            HDF5 group which stores the integrals.
        xints : numpy array, dim: (MO at k point, all MO, all MO)
            extra exchange integrals <pi|iq>_x
        ki : int
            k index for orb i in <pi|iq>_x
        nkpts : int
            Number of k points
        nor : int
            Number of orbitals at a k point.

    Kwargs:
        tol : float, optional
            Below this value integral is not written to file.
            The default is TOL.
    '''

    nspat = len(mapping)
    nuniq = nspat * (nspat + 1) // 2
    # Are the H5 groups already defined? 
    if 'additional_exchange_ints_ispin01' in group_integrals:
        exchange_ints_imag = group_integrals['additional_exchange_ints_im_ispin01']
        exchange_ints_real = group_integrals['additional_exchange_ints_ispin01']
    else:
        exchange_ints_imag = group_integrals.create_dataset('additional_exchange_ints_im_ispin01', shape=(nuniq, nspat), dtype=numpy.float64)
        exchange_ints_real = group_integrals.create_dataset('additional_exchange_ints_ispin01', shape=(nuniq, nspat), dtype=numpy.float64)
    
    for kj in range(nkpts):
        for kk in range(nkpts):
            for i in range(nor):
                for j in range(nor):
                    for k in range(nor):
                        v = xints[i, kj*nor+j, kk*nor+k]/nkpts
                        index_tri_ind, index_repeat = get_hande_index_exchange(i, j, k, i, ki, kj, kk, ki, nor, mapping)
                        if abs(v) > tol:
                            exchange_ints_real[index_tri_ind, index_repeat] = v.real
                            exchange_ints_imag[index_tri_ind, index_repeat] = v.imag

def write_hcore(fout, h, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    '''Write the <i|h|j> integrals to FCIDUMP file.

    Note that <i|h|j> == 0 unless i and j are on the same k point.

    Args:
        fout : file
            FCIDUMP file
        h : array with dimensions (nkpoints, nmo, nmo) [todo] - check
            The <i|h|j> integrals for each kpoint.

    Kwargs:
        tol : float, optional
            Below this value integral is not written to file.
            The default is TOL.
        float_format : str, optional
            Format of integral for writing.
            The default is DEFAULT_FLOAT_FORMAT.
    '''
    # [todo] - is something like hk.reshape(nmo, nmo) required?
    output_format = float_format + ' %4d %4d  0  0\n'
    nmos = 1
    for hk in h:
        nmo = hk.shape[0]
        for i in range(nmo):
            for j in range(0, i+1):
                if abs(hk[i, j]) > tol:
                    fout.write(output_format %
                               (hk[i, j].real, hk[i, j].imag, i+nmos, j+nmos))
        nmos += nmo


def from_integrals(output, h1e, h2e, nmo, nelec, kconserv, nuc, ms, nprop,
                   npropbitlen, orbsym=None, tol=TOL,
                   float_format=DEFAULT_FLOAT_FORMAT):
    '''Use passed in integrals to write integrals to FCIDUMP.

    Args:
        output : str
            Name of FCIDUMP file.
        h1e : array with dimensions (nkpoints, nmo, nmo) [todo] - check
            The <i|h|j> integrals for each k point.
        h2e : numpy array
            Contains ERIs divided by number of k points, with indices
            [ka,kc,kb,a,c,b,d].
        nmo : int
            Number of (molecular) orbitals (sum over all kpoints).
        nelec : int or List of ints [todo]
            Number of electrons.
        kconserv : function
            Pass in three k point indices and get fourth one where
            overall momentum is conserved.

        nuc : float
            Constant, nuclear energy.  Note that it is scaled by number
            of k points.
        ms : int
            Overall spin.
        nprop : List of int
            Number of k points in each dimension.  [todo] - check
        propbitlen : int
            Number of bits in an element of orbsym corresponding to
            a dimension of k.  It means that an element in orbsym
            (which is an integer) can represent k (which is a list of
            integers).  If k is three dimensional, each third of the
            bit representation of an element of orbsym belongs to an
            an element of k.  The lengths of this third, in number of
            bits, is propbitlen.
            [todo] - improve? HANDE has an example (by Charlie)
    Kwargs:
        orbsym : List of ints, optional
            Integers labelling symmetry of the orbitals.
            Default is None, in which case write_head function will
            assign all orbitals the same symmetry, 1.
        tol : float, optional
            Below this value integral is not written to file.
            The default is TOL.
        float_format : str, optional
            Format of integral for writing.
            The default is DEFAULT_FLOAT_FORMAT.
    '''
    with open(output, 'w') as fout:
        write_head(fout, nmo, nelec, ms, nprop, npropbitlen, orbsym=orbsym)
        write_eri(fout, h2e, kconserv, tol=tol, float_format=float_format)
        write_hcore(fout, h1e, tol=tol, float_format=float_format)
        output_format = float_format + '  0  0  0  0\n'
        fout.write(output_format % (nuc, 0))


def _partition(part, rank, nmo, size, nkpts, ntot):  # p
    rem = nmo % size  # p
    dspls = [0]*(size+1)  # p
    for s in range(size):  # p
        if (rem != 0) and ((size - s) <= rem):  # p
            dspls[s+1] = dspls[s]+part+1  # p
        else:  # p
            dspls[s+1] = dspls[s]+part  # p
    l = range(dspls[rank], dspls[rank+1])  # p
    nar = len(l)  # p
    counts = [0]*size  # p
    for s in range(size+1):  # p
        dspls[s] = dspls[s]*2*ntot*ntot  # p
    for s in range(size):  # p
        counts[s] = dspls[s+1]-dspls[s]  # p
    dspls.pop(-1)  # p
    return nar, l, dspls, counts  # p


def exchange_integrals(comm, mf, nmo, kconserv, fout, kstart, kpts, group_integrals=None, mapping=None, HDF5=False):
    '''Calculate <pi|iq>_x exchange integrals.

    Args:
        comm : MPI.COMM_WORLD or None
            Needed for MPI parallelism.
        mf : SCF calculation object
            Stores SCF calculation results.
        nmo : int
            Number of molecular orbital on a k point.
        kconserv : function
            Pass in three k point indices and get fourth one where
            overall momentum is conserved.
        fout : file
            FCIDUMP file
        kstart : int
            Next k point to calculation exchange integral for.
            (Calculation can take time, so this means that this can be
            split)
        kpts : (nkpts, 3) ndarray of floats
            Were found using cell.get_abs_kpts(scaled_kpts).
            Absolute k-points in 1/Bohr.
            [todo]
        group_integrals : ? 
            HDF5 file group to which the above integrals are written.
        mapping : Dict 
            Map between PySCF basis ordering and HANDE basis ordering.
        HDF5 : bool 
            Whether or not to also write the exchange integrals to HANDE 
            compatible .H5 file.
    '''
    if comm == None:  # P
        rank = 0
        size = 1
    else:  # P
        rank = comm.Get_rank()  # P
        size = comm.Get_size()  # P
    nkpts = len(kpts)
    ntot = nkpts*nmo
    if nmo < size:
        print("More processes than molecular orbitals")
    part = nmo // size
    nar, l, dspls, counts = _partition(part, rank, nmo, size, nkpts, ntot)
    xints_p = numpy.ndarray((nar, ntot, ntot), dtype=complex)
    xints_p.fill(0)
    # We wish to calculate <ip|qi> for all i,p,q.  We do this by iterating over
    # all possible orbitals i, and for each, create a density matrix.
    # From this density we create an exchange potential (v_k(G)), which is evaluated at
    # all G vectors.  To form <ip|qi> we calculate the codensity
    # of p and q (again in G space), and integrate (sum) this with the exchange
    # potential of i. i, p and q are different at each k-point, and so we need
    # to loop over k-points for them too, with the proviso of momentum
    # conservation, which is determined by the kconverv table.
    for ik in range(kstart, nkpts):
        for i in l:
            occ = numpy.zeros(nmo)
            occ[i] = 1
            occk = [numpy.zeros(nmo)]*nkpts
            occk[ik] = occ         # just occupy orbital i in kpoint ik
            # According to docs, dm_kpts is a (nkpts, nao, nao) ndarray.
            # dm_kpts is zero at all k points, except for k = ik case
            # where its (nao, nao) sized element contains C^i_m C^{*i}_n
            # entry at position mn.  C^i_m is the coefficient MO i has
            # at position m in AO space. (i, p, q are MO labels, m, n are
            # AO labels here).
            dm_kpts = mf.make_rdm1(mf.mo_coeff, occk)
            # 0 => might not be hermitian
            vk = mf.get_k(cell=None, dm_kpts=dm_kpts, hermi=0, kpts=kpts,
                          kpts_band=None, omega=None)
            # NB that vk is the exchange potential of electron ik,i, expressed
            # in the Bloch-AO basis and in general has components for all k
            # values.
            # [todo] - check:
            # vk is related to K in equation 11 of
            # McClain et al. JCTC, 2017, 13, 3, 1209–1218 but includes
            # contracted molecular coefficients of ik, i.
            for p in range(nmo):
                for pk in range(nkpts):
                    for q in range(nmo):
                        # ks is the (single) allowes k-point conserving symmetry
                        # Since we want (iq|pi) = <ip|qi> = <pi|iq>, we use the
                        # following lookup
                        # Might need to check the ordering of this!
                        # It's definitely elec1,elec1,elec2 but whether the
                        # result is bra or ket is uncertain
                        qk = kconserv[pk, ik, ik]
                        if (qk != pk):
                            print("Error in exchange integrals: k point ",
                                  qk, "is matched to ", pk)
                        # [todo] - check order and einsum order here?
                        # dm_{mn} element is equal to C^{*p}_m C^q_n.
                        dm = numpy.outer(
                            mf.mo_coeff[pk].T[p].T.conj(),
                            mf.mo_coeff[qk].T[q])
                        intgrl = numpy.einsum('ij,ij', dm, vk[pk])
                        # nkpts needed for renormalization so integrals are for
                        # whole supercell not just unit cell
                        xints_p[i-l[0], qk*nmo+q, pk*nmo+p] = intgrl*nkpts
        if comm == None:  # P
            xints = xints_p
        else:  # P
            xints = numpy.ndarray(nmo*ntot*ntot, dtype=complex)  # P
            xints.fill(0)  # P
            comm.Gatherv([xints_p.flatten(), counts[rank], MPI.COMPLEX],  # P
                         [xints, tuple(counts), tuple(
                             dspls), MPI.COMPLEX],  # P
                         root=0)  # P
        if rank == 0:
            xints = xints.reshape((nmo, ntot, ntot))
            if fout is not None:
                write_exchange_integrals(fout, xints, ik, nkpts, nmo)
                fout.flush()
            if HDF5: 
                write_exchange_integrals_HDF5(group_integrals, xints, ik, nkpts, nmo, mapping)

def insertion_rank(arr, tol=1e-12):
    n = len(arr)
    rank = list(range(n))
    for i in range(1, n):
        tmp = rank[i]
        j = i - 1
        while j >= 0 and (arr[rank[j]] - arr[tmp]) >= tol:
            rank[j+1] = rank[j]
            j -= 1
        rank[j+1] = tmp
    return rank  

def fcidump(fcid, mf, kgrid, scaled_kpts_in, MP, keep_exxdiv=False, resume=False,
            parallel=None, HDF5=False):
    '''Dump constant term, orb energies, 1-e and 2-e integrals to file.

    Args:
        fcid : str
            Name of the file to write to. Exchange integrals will be
            written to fcid_X, the rest to fcid.
        mf : SCF calculation object
            Stores SCF calculation results.
        scaled_kpts_in : Numpy array of floats
            Scaled k-points, i.e. (.5,.5,.5) is at the very corner of
            the BZ.
        MP : bool
            True if the grid used is a Monkhorst-Pack grid.
    Kwargs:
        keep_exxdiv : bool
            If True, keep exxdiv treatment used for SCF calculation
            when evaluating two electron integrals.
            [todo] - check! Should this be an option?
            The default is False.
        resume : bool
            If True, the program will check for existing fcid_X file
            and resume the dumping of X integrals.
            The default is False.
        parallel : bool
            If True, MPI parallelization will be used.
            The default is False.
        HDF5 : bool 
            If true, file written will be a suitable HANDE system HDF5 file 
            rather than plain text FCIDUMP.
    '''

    if parallel:  # P
        if not mpi4py_avail:  # P
            raise ImportError("Chosen parallel but mpi4py not available!")  # P
        comm = MPI.COMM_WORLD  # P
    else:  # P
        comm = None  # P
    # scaled_kpts will be modified later so make copy.
    # [todo] - more deepcopying needed?
    scaled_kpts = copy.deepcopy(scaled_kpts_in)
    dummy_cc = KRCCSD(mf)
    # If keep_exxdiv, keeping exxdiv used in scf calculation.
    dummy_cc.keep_exxdiv = keep_exxdiv
    nprop = list(kgrid)

    if comm == None:  # P
        rank = 0
    else:  # P
        rank = comm.Get_rank()  # P
    kconserv = dummy_cc.khelper.kconserv
    nmo = len(mf.mo_coeff[0])
    kps = len(mf.mo_coeff)
    fx = None
    kstart = None
    if rank == 0:
        if not HDF5:
            if resume:
                fx = open(fcid+"_X", 'r')
                lines = fx.readlines()
                line = lines[-1].split()
                kstart = (int(line[1])-1)/nmo + 1
                print("Resuming dumping, starting at k point " + str(kstart))
                fx.close()
                fx = open(fcid+"_X", 'a')
            else:
                fx = open(fcid+"_X", 'w')
                kstart = 0
        else:
            kstart = 0
        npropbitlen = 8
        for i, nk in enumerate(nprop):
            for j in range(scaled_kpts.shape[0]):
                scaled_kpts[j, i] = int(round(scaled_kpts[j, i]*nk)) % nk
        # For each k-point, get the G-space core hamiltonian, and transform it
        # into the molecular orbital basis.
        # Different k-points don't couple.
        # h1es will contain a list with an MOxMO matrix for each k-point.
        h1es = [reduce(numpy.dot,
                       (numpy.asarray(mf.mo_coeff)[k].T.conj(),
                        mf.get_hcore()[k], numpy.asarray(mf.mo_coeff)[k]))
                for k in range(kps)]
    if comm != None:
        kstart = comm.bcast(kstart, root=0)
    if HDF5:
        if rank == 0:
            with h5py.File(fcid + '.H5', 'w') as f:
                group_system = f.create_group('/system')
                group_read_in = group_system.create_group('read_in')
                group_integrals = group_read_in.create_group('integrals')
                # Sort spin orbitals by global energy ordering (HANDE convention).
                orbs = []
                original_index = 0
                for k_index in range(kps):
                    for p, e in enumerate(mf.mo_energy[k_index]):
                        orbs.append((e, k_index, scaled_kpts[k_index], p, original_index))
                        original_index += 1
                energies = [e.real for k in mf.mo_energy for e in k]             
                ranking = insertion_rank(energies, tol=1e-12)
                orbs = [orbs[i] for i in ranking]
                mapping = {}
                for spatial_idx, (_, _, _, _, orig) in enumerate(orbs):
                    mapping[orig] = spatial_idx # Mapping from the original k-point grouped ordering to HANDEs global energy ordering of spatial orbitals
                spin_orbs = []
                for i, (e, k, k_vec, p, original_index) in enumerate(orbs, start = 1):
                    # Spin orbitals may be constructed in this way as we consider only RHF for now.
                    spin_orbs.append(dict(spin_idx = 2 * i - 1, spin = 1, spatial_idx = i, energy = e, k = k, k_vec = k_vec, p = p))
                    spin_orbs.append(dict(spin_idx = 2 * i, spin = -1, spatial_idx = i, energy = e, k = k, k_vec = k_vec, p = p))
                nbasis = numpy.int32(len(spin_orbs))
                exchange_integrals(comm, mf, nmo, kconserv, None, kstart, mf.kpts, group_integrals=group_integrals, mapping=mapping, HDF5=HDF5)
        else:
                exchange_integrals(comm, mf, nmo, kconserv, None, kstart, mf.kpts, group_integrals=None, mapping=None, HDF5=True)
    else:
        exchange_integrals(comm, mf, nmo, kconserv, fx, kstart, mf.kpts, group_integrals=None, mapping=None, HDF5=False)
    if rank == 0:
        if not HDF5:
            fx.close()
        # MP meshes with an even number of points in a dimension do not contain
        # the Gamma point.
        # Unfortunately this is not compatible with some symmetry
        # specifications, so if we multiply that dimension's kpoint grid by 2,
        # we get a grid which can contain both the MP mesh and the Gamma point
        # (even though we don't have any actual orbitals calculated at the
        # gamma point).
        if MP:
            for i in range(3):
                if nprop[i] % 2 == 0:
                    nprop[i] *= 2
        eris = dummy_cc.ao2mo()
        nel = sum(sum(mf.mo_occ))
        orbsym = []
        propsc = 2**npropbitlen
        for k in range(kps):
            n = scaled_kpts[k, 0]+propsc*scaled_kpts[k, 1] + \
                propsc*propsc*scaled_kpts[k, 2]
            orbsym += [int(n)]*nmo
        nkpts = kgrid[0]*kgrid[1]*kgrid[2]
        if HDF5:
            with h5py.File(fcid + '.H5', 'a') as f:
                group_metadata = f.create_group('/metadata')
                timestamp = datetime.datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                hande_version = '0000000000000000000000000000000000000000' # Dummy value 
                sysdump_version = 0 # Dummy value 
                ascii19 = h5py.string_dtype(encoding='ascii', length=19)
                ascii36 = h5py.string_dtype(encoding='ascii', length=36)
                ascii40 = h5py.string_dtype(encoding='ascii', length=40)
                ascii255 = h5py.string_dtype(encoding='ascii', length=255)
                group_metadata.create_dataset('date', data=numpy.array(timestamp, dtype=ascii19))
                group_metadata.create_dataset('hande version', data=numpy.array(hande_version, dtype=ascii40))
                group_metadata.create_dataset('sysdump version', data=numpy.int32(sysdump_version))
                group_metadata.create_dataset('uuid', data=numpy.array(str(uuid.uuid4()), dtype=ascii36))
                group_system = f['/system']
                group_read_in = group_system['read_in']
                group_integrals = group_read_in['integrals']
                CAS = numpy.array([-1, -1]) # Can define CAS here if need be, [-1, -1] indicates use of all space. 
                Ms = 0 # Ms is 0 for RHF.
                group_system.create_dataset('CAS', data=CAS, dtype=numpy.int32)
                group_system.create_dataset('Ms', data=Ms, dtype=numpy.int32)
                basis_l_numbers = numpy.zeros((3, nbasis), dtype=numpy.int32)
                for col, orb in enumerate(spin_orbs):
                    kvec = scaled_kpts[orb['k']]
                    basis_l_numbers[:, col] = kvec
                basis_lz = numpy.zeros((nbasis), dtype=numpy.int32) # FCIDUMP orbitals do not commute with Lz --> ignore this symmetry.
                basis_ms = numpy.array([orb['spin'] for orb in spin_orbs], dtype=numpy.int32)
                basis_sp_eigv = numpy.array([orb['energy'] for orb in spin_orbs], dtype=numpy.float64)
                basis_spatial_index = numpy.array([orb['spatial_idx'] for orb in spin_orbs], dtype=numpy.int32)
                basis_symmetry = numpy.array([1 + scaled_kpts[orb['k']][0] + nprop[0] * scaled_kpts[orb['k']][1] 
                                                + nprop[0] * nprop[1] * scaled_kpts[orb['k']][2] for orb in spin_orbs], dtype=numpy.int32)
                basis_symmetry_index = numpy.zeros((nbasis), dtype=numpy.int32) # Only used for point group symmetry i.e., not periodic systems
                basis_symmetry_spin_index = numpy.zeros((nbasis), dtype=numpy.int32)
                counter = {}
                for i, orb in enumerate(spin_orbs):
                    key = (basis_symmetry[i], orb['spin'])
                    n = counter.get(key, 1)
                    basis_symmetry_spin_index[i] = n 
                    counter[key] = n + 1
                group_basis = group_system.create_group('basis')
                group_basis.create_dataset('basis_l_numbers', data=basis_l_numbers)
                group_basis.create_dataset('basis_lz', data=basis_lz)
                group_basis.create_dataset('basis_ms', data=basis_ms)
                group_basis.create_dataset('basis_sp_eigv', data=basis_sp_eigv)
                group_basis.create_dataset('basis_spatial_index', data=basis_spatial_index)
                group_basis.create_dataset('basis_symmetry', data=basis_symmetry)
                group_basis.create_dataset('basis_symmetry_index', data=basis_symmetry_index)
                group_basis.create_dataset('basis_symmetry_spin_index', data=basis_symmetry_spin_index)
                group_basis.create_dataset('nbasis', data=nbasis)
                group_system.create_dataset('momentum_space', data=numpy.int32(1)) # 1 for true 
                group_system.create_dataset('nelectrons', data=numpy.int32(nel))
                ecore_val = numpy.array([nkpts * mf.mol.energy_nuc()])
                group_read_in.create_dataset('comp', data=numpy.int32(1)) # eris will be complex valued 
                group_read_in.create_dataset('ecore', data=numpy.array(ecore_val, dtype=numpy.float64))
                group_read_in.create_dataset('ex_exchange_ints', data=numpy.int32(1))  # Exchange ints written
                group_read_in.create_dataset('fcidump', data=numpy.array(fcid, dtype=ascii255))
                group_read_in.create_dataset('ex_fcidump', data=numpy.array(fcid + '_X', dtype=ascii255))
                write_eri_HDF5(group_integrals, eris, kconserv, mapping, tol=TOL)
                xints_real = f['/system/read_in/integrals/additional_exchange_ints_ispin01']
                xints_imag = f['/system/read_in/integrals/additional_exchange_ints_im_ispin01']
                coulomb_ints_real = f['/system/read_in/integrals/coulomb_ints_ispin01']
                coulomb_ints_imag = f['/system/read_in/integrals/coulomb_ints_im_ispin01']
                occ_k = [[i for i,occ in enumerate(mf.mo_occ[k]) if occ > 1e-8] for k in range(kps)]
                for k in range(kps):
                    H = h1es[k].astype(numpy.complex128)
                    delta = numpy.zeros_like(H, dtype=numpy.complex128)
                    for a in range(nmo):
                        for b in range(nmo):
                            JR = JI = KR = KI = 0.0
                            for ki in range(kps):
                                for i in range(nmo):
                                    j_idx = get_hande_index_coulomb(i, b, a, i, ki, k, k, ki, nmo, mapping)
                                    JR += coulomb_ints_real[j_idx]
                                    JI += coulomb_ints_imag[j_idx]
                                    tri, rep = get_hande_index_exchange(i, b, a, i, ki, k, k, ki, nmo, mapping)
                                    KR += xints_real[tri, rep]
                                    KI += xints_imag[tri, rep]
                            delta[a, b] = -0.5*((KR - JR) + 1j*(KI - JI))
                    H += delta 
                    isym = int(1 + (scaled_kpts[k,0]) + nprop[0]*scaled_kpts[k,1] + nprop[0]*nprop[1]*scaled_kpts[k,2])
                    re_upper = [] 
                    im_upper = []
                    for j in range(nmo):
                        for i in range(j + 1):
                            hij = H[i, j]
                            re_upper.append(0.0 if abs(hij.real) < TOL else hij.real)
                            im_upper.append(hij.imag)
                    group_integrals.create_dataset(f'one_body_ispin01_isym{isym:02d}', data=numpy.asarray(re_upper, dtype=numpy.float64))
                    group_integrals.create_dataset(f'one_body_im_ispin01_isym{isym:02d}', data=numpy.array(im_upper, dtype=numpy.float64))
                group_read_in.create_dataset('nprop', data=numpy.array(nprop, dtype=numpy.int32))
                group_read_in.create_dataset('uhf', data=numpy.int32(0)) # Only considering RHF 
                group_read_in.create_dataset('uselz', data=numpy.int32(0)) # Not using Lz symmetry
                group_system.create_dataset('system', data=numpy.int32(2)) # read_in enum parameter in HANDE is 2
        if not HDF5:
            from_integrals(fcid, h1es, eris, kps*nmo, nel, kconserv,
                           nkpts*mf.mol.energy_nuc(), 0, nprop, npropbitlen,
                           orbsym=orbsym)
            # Write orbital energies to fcid file, too.
            f = open(fcid, 'a')
            n = 0
            for k in range(kps):
                for e in mf.mo_energy[k]:
                    n += 1
                    f.write(' (%.16g,%.16g) %4d %4d %4d %4d\n' %
                            (e.real, e.imag, n, 0, 0, 0))
            f.close()
