import json
import scipy
from typing import Dict, Any

def ci_95(ac, an):
    if ac == 0:
        return 0

    if ac > 0:
        af = ac/an

        # 95% confidence interval is caluclated from the upper bound of the chi_squared approximation.
        # This is the filtering allele frequency (FAF) calculated from a Poisson distribution.
        # Recommended by Whiffin et al. (2017)
        chi_squqred = scipy.stats.chi2.ppf(0.95, 2*(ac + 1))
        upper = (chi_squqred / 2) / an
        return {'af': af, 'upper': upper}

ci = ci_95(42, 1179700)
print(ci['upper'])


class FHVariantToolkit:
    """
    A specialized toolkit for calculating Allele Frequency (AF) thresholds
    based on ClinGen FH-VCEP (v1) and ACGS (2024) guidelines.
    """

    def __init__(self, gene: str = "LDLR"):
        self.gene = gene
        # FH-VCEP specifically lowers BA1 for LDLR from 5% to 1%
        # PM2 is often refined to 'Absent' or specific PopMax thresholds
        self.thresholds = {
            "BA1": 0.01,  # Benign Stand-alone (1%)
            "BS1": 0.005,  # Benign Strong (0.5%)
            "PM2": 0.0002  # Pathogenic Moderate (0.02%)
        }

    def evaluate_frequency_criteria(self, json_input: str) -> Dict[str, Any]:
        """
        Calculates which ACMG/ClinGen frequency criteria are met.
        Input: JSON string with 'pop_max_af' (gnomAD PopMax/GrpMax).
        """
        try:
            data = json.loads(json_input)
            af = data.get("pop_max_af", 0.0)
            results = {
                "variant": data.get("nomenclature", "Unknown"),
                "applied_criteria": [],
                "reasoning": ""
            }

            # Apply Logic for LDLR (FH-VCEP)
            if af >= self.thresholds["BA1"]:
                results["applied_criteria"].append({"code": "BA1", "strength": "Stand-alone"})
            elif af >= self.thresholds["BS1"]:
                results["applied_criteria"].append({"code": "BS1", "strength": "Strong"})

            # ACGS 2024 emphasizes PM2 as 'Supporting' if very rare/absent
            if af == 0 or af <= self.thresholds["PM2"]:
                results["applied_criteria"].append({"code": "PM2", "strength": "Moderate"})

            results["reasoning"] = f"PopMax AF of {af} evaluated against {self.gene} thresholds."
            return results

        except json.JSONDecodeError:
            return {"error": "Invalid JSON input provided to toolkit."}


# Example of an Agent's tool call:
toolkit = FHVariantToolkit(gene="LDLR")
sample_data = '{"nomenclature": "LDLR:c.301G>A", "pop_max_af": 0.015}'
analysis = toolkit.evaluate_frequency_criteria(sample_data)

print(json.dumps(analysis, indent=2))


gene_symbol = 'LDLR'
variant = 'c.301G>A'

