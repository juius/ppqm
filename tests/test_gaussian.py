from collections import ChainMap

import pytest
from context import G16_OPTIONS, RESOURCES

from ppqm import chembridge, gaussian, tasks

TEST_ENERGIES = [
    ("O", -51.76150372010862),
    ("CC", -15.49718624442594),
    ("[NH4+]", 156.64505678634524),
]

logfile_optfreq = RESOURCES / "gaussian/cnh5_optfreq.out"
with open(logfile_optfreq) as f:
    lines_optfreq = f.readlines()

def _get_options(tmpdir):
    g16_options = {"scr": tmpdir, "memory": 2}
    options_prime = ChainMap(g16_options, G16_OPTIONS)
    options_prime = dict(options_prime)
    return options_prime

def test_get_optimized_structure():
    properties = gaussian.get_opt_structure(lines_optfreq)
    atoms = properties[gaussian.COLUMN_ATOMS]
    assert (atoms == np.array([6, 1, 1, 1, 7, 1, 1], dtype=int)).all()
    assert properties[gaussian.COLUMN_COORD] is not None
    
def test_get_frequencies():
    properties = gaussian.get_frequencies(lines_optfreq)
    assert properties[gaussian.COLUMN_FREQUENCIES] is not None
    assert properties[gaussian.COLUMN_FREQUENCIES][0] == 327.696

    assert properties[gaussian.COLUMN_NORMAL_COORD] is not None
    assert len(properties[gaussian.COLUMN_NORMAL_COORD][0]) == 7

@pytest.mark.parametrize("smiles, energy", TEST_ENERGIES)
def test_axyzc_optimize(smiles, energy, tmpdir):
    """ TODO: optimize not just SP """

    g16_options = _get_options(tmpdir)

    molobj = chembridge.smiles_to_molobj(smiles)
    molobj = tasks.generate_conformers(molobj, n_conformers=1)

    assert molobj is not None

    calculation_options = {"pm3": None}
    atoms, coordinates, charge = chembridge.get_axyzc(molobj, atomfmt=str)

    properties = gaussian.get_properties_from_axyzc(
        atoms, coordinates, charge, spin=1, options=calculation_options, **g16_options
    )

    scf_energy = properties[gaussian.COLUMN_SCF_ENERGY]
    assert pytest.approx(energy, 10 ** -4) == scf_energy


def test_parse_mulliken_charges():
    pass


def test_parse_hirshfeld_charges():

    logfilename = RESOURCES / "gaussian/cnh5_pop.out"
    with open(logfilename) as f:
        lines = f.readlines()

    hirshfeld_charges = gaussian.get_hirsfeld_charges(lines)

    assert "hirshfeld_charges" in hirshfeld_charges
    assert len(hirshfeld_charges["hirshfeld_charges"]) == 7

    assert isinstance(hirshfeld_charges["hirshfeld_charges"][0], float)
    assert hirshfeld_charges["hirshfeld_charges"][0] == -0.034338
    assert hirshfeld_charges["hirshfeld_charges"][-1] == 0.094812

    assert round(sum(hirshfeld_charges["hirshfeld_charges"]), 5) == 0.0


def test_parse_cm5_charges():

    logfilename = RESOURCES / "gaussian/cnh5_pop.out"
    with open(logfilename) as f:
        lines = f.readlines()

    cm5_charges = gaussian.get_cm5_charges(lines)

    assert "cm5_charges" in cm5_charges
    assert len(cm5_charges["cm5_charges"]) == 7

    assert isinstance(cm5_charges["cm5_charges"][0], float)
    assert cm5_charges["cm5_charges"][0] == -0.139922
    assert cm5_charges["cm5_charges"][-1] == 0.287584

    assert round(sum(cm5_charges["cm5_charges"]), 5) == 0.0


def test_nbo_bond_orders():

    logfilename = RESOURCES / "gaussian/c3nh11_extra.out"
    with open(logfilename) as f:
        lines = f.readlines()

    nbo_bond_order = gaussian.get_nbo_bond_orders(lines)

    assert "bond_orders" in nbo_bond_order

    assert len(nbo_bond_order["bond_orders"]) == 13
    assert len(nbo_bond_order["bond_orders"][0]) == 13

    # TODO: Check that the values are set the correct places.


def test_nmr_shielding_constants():

    logfilename = RESOURCES / "gaussian/cnh5_nmr.out"
    with open(logfilename) as f:
        lines = f.readlines()

    shielding_constants = gaussian.get_nmr_shielding_constants(lines)

    assert "shielding_constants" in shielding_constants

    assert len(shielding_constants["shielding_constants"]) == 7
    assert isinstance(shielding_constants["shielding_constants"][0], float)

    assert shielding_constants["shielding_constants"][0] == 165.1796
    assert shielding_constants["shielding_constants"][-1] == 32.4508
