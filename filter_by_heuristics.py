import json
import os
import signal
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import MCS
from tqdm import tqdm
RDLogger.DisableLog("rdApp.*")


def canonicalize_smiles(smi: str, useChiral: bool):
    """Adapted from Molecular Transformer"""
    smiles = "".join(smi.split())
    cano_smiles = ""

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        frags = Chem.GetMolFrags(mol, asMols=True)
        num_atoms = [m.GetNumAtoms() for m in frags]
        idx = np.argmax(num_atoms)
        mol = frags[idx]
        cano_smiles = Chem.MolToSmiles(mol, isomericSmiles=useChiral, canonical=True)
        # Sometimes stereochemistry takes another canonicalization... (just in case)
        mol = Chem.MolFromSmiles(cano_smiles)
        if mol is not None:
            cano_smiles = Chem.MolToSmiles(mol, isomericSmiles=useChiral, canonical=True)

    return cano_smiles


def find_mcs_with_timeout(mols, timeout_duration=1):
    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_duration)
    try:
        result = MCS.FindMCS(mols)
    except TimeoutError as exc:
        result = None
    finally:
        signal.alarm(0)

    return result


def is_valid_by_identity(reaction):
    r_smis = [canonicalize_smiles(reactant["smiles"], useChiral=True)
              for reactant in reaction["reactants"]
              if reactant["category_id"] == 1]
    p_smis = [canonicalize_smiles(product["smiles"], useChiral=True)
              for product in reaction["products"]
              if product["category_id"] == 1]

    r_smis = set([smi for smi in r_smis if smi])
    p_smis = set([smi for smi in p_smis if smi])

    if len(reaction["conditions"]) > 0 or len(r_smis) > 1 or len(p_smis) > 1:
        return True
    if any(smi in r_smis for smi in p_smis):
        print(f"Not valid by identity, r: {r_smis}, p: {p_smis}")
        return False

    return True


def is_valid_by_mcs(reaction, min_bonds: int):
    r_smis = [canonicalize_smiles(reactant["smiles"], useChiral=True)
              for reactant in reaction["reactants"]
              if reactant["category_id"] == 1]
    p_smis = [canonicalize_smiles(product["smiles"], useChiral=True)
              for product in reaction["products"]
              if product["category_id"] == 1]

    r_smis = set([smi for smi in r_smis if smi])
    p_smis = set([smi for smi in p_smis if smi])

    if len(reaction["conditions"]) > 0 or len(r_smis) > 1 or len(p_smis) > 1:
        return True
    if len(r_smis) == 0 and len(p_smis) == 0:
        return True
    if len(r_smis) == 0 or len(p_smis) == 0:        # placeholder
        return True
    if any('*' in smi for smi in r_smis) or any('*' in smi for smi in p_smis):
        return True

    mcs_result = find_mcs_with_timeout(mols=[Chem.MolFromSmiles(list(r_smis)[0]),
                                             Chem.MolFromSmiles(list(p_smis)[0])],
                                       timeout_duration=1)
    if mcs_result is None:
        print(f"Timeout finding MCS, r: {r_smis}, p: {p_smis}")
        return True
    if mcs_result.numBonds < min_bonds:
        print(f"Not valid by MCS, r: {r_smis}, p: {p_smis}")
        return False

    return True


def is_valid_by_conservation(reaction):
    if not all(r["category_id"] == 1 and not r["smiles"] == "<invalid>"
               for r in reaction["reactants"]):
        return True
    if not all(c["category_id"] == 1 and not c["smiles"] == "<invalid>"
               for c in reaction["conditions"]):
        return True
    if not all(p["category_id"] == 1 and not p["smiles"] == "<invalid>"
               for p in reaction["products"]):
        return True

    c_in_r = sum(r["smiles"].count('C') + r["smiles"].count('c')
                 for r in reaction["reactants"])
    c_in_c = sum(c["smiles"].count('C') + c["smiles"].count('c')
                 for c in reaction["conditions"])
    c_in_p = sum(p["smiles"].count('C') + p["smiles"].count('c')
                 for p in reaction["products"])

    max_carbon_diff = 6
    if c_in_r + c_in_c - c_in_p > max_carbon_diff:
        print(f"Not valid by carbon conservation w condition, "
              f"r: {[r['smiles'] for r in reaction['reactants']]}, "
              f"c: {[c['smiles'] for c in reaction['conditions']]}, "
              f"p: {[p['smiles'] for p in reaction['products']]}.")
        return False

    return True


def is_valid_by_ring_count(reaction):
    if not all(r["category_id"] == 1 and not r["smiles"] == "<invalid>"
               for r in reaction["reactants"]):
        return True
    if not all(c["category_id"] == 1 and not c["smiles"] == "<invalid>"
               for c in reaction["conditions"]):
        return True
    if not all(p["category_id"] == 1 and not p["smiles"] == "<invalid>"
               for p in reaction["products"]):
        return True
    r_mols = [Chem.MolFromSmiles(r["smiles"]) for r in reaction["reactants"]]
    c_mols = [Chem.MolFromSmiles(c["smiles"]) for c in reaction["conditions"]]
    p_mols = [Chem.MolFromSmiles(p["smiles"]) for p in reaction["products"]]

    if any(mol is None for mol in r_mols + c_mols + p_mols):
        return True
    ring_in_r = sum(len(mol.GetRingInfo().AtomRings()) for mol in r_mols)
    ring_in_c = sum(len(mol.GetRingInfo().AtomRings()) for mol in c_mols)
    ring_in_p = sum(len(mol.GetRingInfo().AtomRings()) for mol in p_mols)

    max_ring_diff = 2
    if ring_in_r + ring_in_c - ring_in_p > max_ring_diff:
        print(f"Not valid by ring count difference, "
              f"r: {[r['smiles'] for r in reaction['reactants']]}, "
              f"c: {[c['smiles'] for c in reaction['conditions']]}, "
              f"p: {[p['smiles'] for p in reaction['products']]}.")
        return False

    return True


def filter_reactions(fp: str, ofp: str,
                     by_identity: bool, by_mcs: bool,
                     by_conservation: bool,
                     by_ring_count: bool):
    for split in range(5):
        os.makedirs(os.path.join(ofp, f"{split}"), exist_ok=True)
        fn = os.path.join(fp, f"{split}", f"prediction_test{split}.json")
        ofn = os.path.join(ofp, f"{split}", f"filter_test{split}.json")

        with open(fn, "r") as f:
            data = json.load(f)
        reactions_in_order = data["reaction"]
        for doc_idx, reactions_per_doc in enumerate(tqdm(reactions_in_order)):
            filtered_reactions = []
            for rxn_idx, reaction in enumerate(reactions_per_doc):
                if by_identity and not is_valid_by_identity(reaction):
                    continue
                if by_mcs and not is_valid_by_mcs(reaction, min_bonds=2):
                    continue
                if by_conservation and not is_valid_by_conservation(reaction):
                    print(split, doc_idx, rxn_idx)
                    continue
                if by_ring_count and not is_valid_by_ring_count(reaction):
                    continue
                filtered_reactions.append(reaction)
            reactions_in_order[doc_idx] = filtered_reactions

        data["reaction"] = reactions_in_order
        with open(ofn, "w") as of:
            json.dump(data, of, indent=4)


def main():
    fp = "output/pix2seq_reaction_nov_cv"

    filter_reactions(fp, fp,
                     by_identity=False,
                     by_mcs=False,
                     by_conservation=True,
                     by_ring_count=False)


if __name__ == "__main__":
    main()
