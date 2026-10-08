# -*- coding: utf-8 -*-
"""
=====================================================================
GENERADOR DE DATASET SINTÉTICO — SECTOR SEGUROS (Colombia)
=====================================================================
Genera 3 tablas relacionales "sucias" listas para un flujo ETL + ML + Power BI:

    clientes_raw.csv   (1 fila por cliente)
    polizas_raw.csv    (N pólizas por cliente        -> FK: cliente_id)
    reclamos_raw.csv   (N reclamos por póliza        -> FK: poliza_id)

CÓMO ESTÁ CONSTRUIDO (en 3 fases)
  1) Se generan datos LIMPIOS y coherentes (con relaciones y patrones latentes).
  2) Se calculan las variables objetivo con modelos logísticos "ocultos":
       - CHURN : estado == "Cancelada"  (sube con alzas de prima y reclamos rechazados)
       - FRAUDE: sospecha_fraude == 1   (~6.5% de los reclamos)
  3) Se "ensucian" los datos: texto inconsistente, fechas mixtas, nulos,
     outliers, llaves huérfanas y duplicados exactos (2%-5%).

USO: ejecutar completo en Google Colab / Jupyter / terminal.
     Los CSV quedan en OUTPUT_DIR (por defecto, la carpeta actual).

NOTA DE LEAKAGE (para tu fase de ML): las columnas `fecha_cancelacion` y
`motivo_cancelacion` solo existen DESPUÉS de que el cliente cancela. No las
uses como variables predictoras del churn.
=====================================================================
"""

import sys
import subprocess
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from faker import Faker
except ImportError:                                   # Colab suele no traerlo
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "faker"])
    from faker import Faker

# ---------------------------------------------------------------------
# 0. CONFIGURACIÓN
# ---------------------------------------------------------------------
SEED = 42                       # cambia la semilla para obtener otro dataset
N_CLIENTES = 5_200              # filas únicas (el mínimo pedido es 5,000)
N_POLIZAS = 8_400               # filas únicas (mínimo 8,000)
N_RECLAMOS = 4_300              # filas únicas (mínimo 4,000)
FECHA_CORTE = pd.Timestamp("2025-12-31")   # "hoy" dentro del dataset
TASA_CHURN = 0.19               # % objetivo de pólizas canceladas
TASA_FRAUDE = 0.065             # % objetivo de reclamos sospechosos (rango 5-8%)
OUTPUT_DIR = Path(".")
GUARDAR_REFERENCIA_LIMPIA = False   # True: guarda también las tablas sin ensuciar

rng = np.random.default_rng(SEED)
try:
    fake = Faker("es_CO")
except Exception:
    fake = Faker("es_ES")
Faker.seed(SEED)

# ---------------------------------------------------------------------
# 1. CATÁLOGOS
# ---------------------------------------------------------------------
CIUDADES = {"Bogotá": .34, "Medellín": .17, "Cali": .12, "Barranquilla": .08,
            "Bucaramanga": .06, "Cartagena": .06, "Pereira": .05, "Cúcuta": .05,
            "Manizales": .04, "Santa Marta": .03}
OCUPACIONES = ["Empleado", "Independiente", "Empresario", "Pensionado", "Estudiante"]
ESTADOS_CIVILES = ["Soltero", "Casado", "Unión libre", "Divorciado", "Viudo"]
NIVELES_EDU = ["Bachiller", "Técnico", "Profesional", "Posgrado"]
CANALES = ["Agencia", "Web", "Asesor comercial", "Corredor", "Bancaseguros"]
COBERTURAS = ["Auto", "Salud", "Hogar", "Vida"]
P_COBERTURA = [.38, .27, .20, .15]
FORMAS_PAGO = ["Mensual", "Trimestral", "Semestral", "Anual"]

MED_SUMA = {"Auto": 55e6, "Salud": 120e6, "Hogar": 220e6, "Vida": 300e6}
SIG_SUMA = {"Auto": .45, "Salud": .50, "Hogar": .50, "Vida": .60}
TASA_PRIMA = {"Auto": .042, "Salud": .022, "Hogar": .0040, "Vida": .0035}
FREQ_SINIESTRO = {"Auto": 1.0, "Salud": 1.3, "Hogar": .5, "Vida": .15}
MED_RECLAMO = {"Auto": 3.2e6, "Salud": 2.4e6, "Hogar": 3.8e6, "Vida": 45e6}
MULT_TIPO = {("Auto", "Robo"): 8, ("Hogar", "Incendio"): 6, ("Hogar", "Robo"): 2,
             ("Salud", "Cirugía"): 3, ("Salud", "Hospitalización"): 2}
TIPOS_SINIESTRO = {
    "Auto": (["Colisión", "Robo", "Daño a terceros", "Vidrios"], [.42, .12, .30, .16]),
    "Salud": (["Hospitalización", "Cirugía", "Consulta especializada", "Medicamentos"],
              [.18, .14, .40, .28]),
    "Hogar": (["Incendio", "Inundación", "Robo", "Daños eléctricos"], [.08, .22, .25, .45]),
    "Vida": (["Fallecimiento", "Incapacidad", "Enfermedad grave"], [.25, .35, .40]),
}

# Sinónimos "de la vida real" que tendrás que homologar en el ETL
ALIAS_CIUDAD = {"Bogotá": ["Bogotá D.C.", "Bogota DC", "Bta"],
                "Medellín": ["Medellin", "Medellín (Antioquia)"],
                "Cali": ["Santiago de Cali"], "Barranquilla": ["B/quilla"],
                "Cartagena": ["Cartagena de Indias"], "Bucaramanga": ["B/manga"],
                "Santa Marta": ["Sta. Marta"]}
ALIAS_COBERTURA = {"Auto": ["Automóvil", "Vehículo", "Seguro Auto"],
                   "Salud": ["Seguro de Salud"], "Hogar": ["Vivienda", "Seguro Hogar"],
                   "Vida": ["Seguro de Vida"]}
ALIAS_ESTADO_POL = {"Activa": ["Vigente", "Activo"], "Cancelada": ["Cancelado", "Anulada"],
                    "Vencida": ["Expirada"], "Renovada": ["Renovado"],
                    "Suspendida": ["Suspendido"]}
ALIAS_ESTADO_REC = {"Rechazado": ["Rechazada", "Denegado"],
                    "Aprobado": ["Aprobada", "Autorizado"],
                    "En revisión": ["En revision", "Pendiente"],
                    "Pagado parcial": ["Pago parcial"]}
ALIAS_GENERO = {"M": ["Masculino", "masc"], "F": ["Femenino", "fem"]}

FORMATOS_FECHA = ["%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"]
P_FORMATOS_FECHA = [.50, .30, .20]


# ---------------------------------------------------------------------
# 2. UTILIDADES
# ---------------------------------------------------------------------
def sigmoide(x):
    return 1 / (1 + np.exp(-x))


def calibrar_intercepto(logit_sin_b0, tasa_objetivo):
    """Busca (bisección) el intercepto para que la prob. media sea `tasa_objetivo`."""
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sigmoide(logit_sin_b0 + mid).mean() > tasa_objetivo:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def idx_aleatorios(n, p):
    """Posiciones de ~p*n filas elegidas al azar."""
    return np.flatnonzero(rng.random(n) < p)


def quitar_tildes(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


TRANSFORMACIONES = [
    str.lower,
    str.upper,
    lambda s: f" {s} ",
    lambda s: f"{s}  ",
    lambda s: f"  {s}",
    quitar_tildes,
    lambda s: quitar_tildes(s).upper(),
    lambda s: s.replace(" ", "  "),
]


def ensuciar_texto(serie, p=0.30, alias=None, p_alias=0.0):
    """Mayúsculas/minúsculas mixtas, espacios extra, tildes perdidas y sinónimos."""
    vals = serie.astype(object).to_numpy().copy()
    for i, v in enumerate(vals):
        if not isinstance(v, str):
            continue
        if alias and v in alias and rng.random() < p_alias:
            v = alias[v][rng.integers(len(alias[v]))]
        if rng.random() < p:
            v = TRANSFORMACIONES[rng.integers(len(TRANSFORMACIONES))](v)
        vals[i] = v
    return pd.Series(vals, index=serie.index)


def poner_nulos(serie, p):
    """Inserta NaN aleatorios (numéricas pasan a float para admitirlos)."""
    s = serie.astype("float64") if pd.api.types.is_numeric_dtype(serie) else serie.astype(object)
    s.iloc[idx_aleatorios(len(s), p)] = np.nan
    return s


def a_entero_nulo(serie):
    """Entero con nulos permitidos (evita '41.0' en el CSV)."""
    return pd.to_numeric(serie, errors="coerce").round().astype("Int64")


def mezclar_fechas(serie, p_nulo=0.02, p_invalido=0.015):
    """Convierte fechas a texto en 3 formatos distintos + 'INVALID_DATE' + nulos."""
    s = pd.to_datetime(serie)
    out = pd.Series(np.nan, index=s.index, dtype=object)
    eleccion = rng.choice(len(FORMATOS_FECHA), size=len(s), p=P_FORMATOS_FECHA)
    for k, fmt in enumerate(FORMATOS_FECHA):
        m = eleccion == k
        out.loc[m] = s[m].dt.strftime(fmt)
    u = rng.random(len(s))
    out.loc[u < p_nulo] = np.nan
    out.loc[(u >= p_nulo) & (u < p_nulo + p_invalido)] = "INVALID_DATE"
    return out


def agregar_duplicados(df):
    """Duplica exactamente entre 2% y 5% de las filas y baraja el resultado."""
    frac = rng.uniform(0.02, 0.05)
    dup = df.sample(frac=frac, random_state=int(rng.integers(1_000_000_000)))
    return (pd.concat([df, dup])
            .sample(frac=1, random_state=int(rng.integers(1_000_000_000)))
            .reset_index(drop=True))


# ---------------------------------------------------------------------
# 3. GENERACIÓN DE DATOS LIMPIOS
# ---------------------------------------------------------------------
def generar_clientes():
    n = N_CLIENTES
    genero = rng.choice(["M", "F"], n, p=[.48, .52])
    edad = np.clip(rng.normal(41, 13, n).round(), 18, 80).astype(int)
    ciudad = rng.choice(list(CIUDADES), n, p=list(CIUDADES.values()))
    estrato = rng.choice([1, 2, 3, 4, 5, 6], n, p=[.12, .30, .30, .15, .08, .05])

    ocup = rng.choice(OCUPACIONES, n, p=[.46, .28, .10, .11, .05])
    ocup = np.where((edad >= 60) & (rng.random(n) < .7), "Pensionado", ocup)
    ocup = np.where((edad > 30) & (ocup == "Estudiante"), "Empleado", ocup)

    base = pd.Series(estrato).map({1: 1.6e6, 2: 2.2e6, 3: 3.4e6, 4: 5.5e6,
                                   5: 8.5e6, 6: 14e6}).to_numpy()
    f_ocup = pd.Series(ocup).map({"Empleado": 1, "Independiente": .95, "Empresario": 1.8,
                                  "Pensionado": .85, "Estudiante": .4}).to_numpy()
    ingreso = np.round(base * f_ocup * rng.lognormal(0, .35, n), -3)

    # El score crece con el ingreso y con la edad (más historial crediticio)
    z_ing = (np.log(ingreso) - np.log(3.4e6)) / .7
    score = np.clip(640 + 45 * z_ing + 0.8 * (edad - 41) + rng.normal(0, 70, n),
                    300, 950).round().astype(int)

    dias_reg = (pd.Timestamp("2024-12-31") - pd.Timestamp("2017-01-01")).days + 1
    f_registro = pd.Timestamp("2017-01-01") + pd.to_timedelta(rng.integers(0, dias_reg, n), unit="D")

    return pd.DataFrame({
        "cliente_id": [f"CLI-{i:05d}" for i in range(1, n + 1)],
        "nombre": [fake.name() for _ in range(n)],
        "documento": rng.integers(1_000_000, 1_900_000_000, n),
        "genero": genero,
        "edad": edad,
        "ciudad": ciudad,
        "estrato": estrato,
        "estado_civil": rng.choice(ESTADOS_CIVILES, n, p=[.35, .30, .20, .10, .05]),
        "nivel_educativo": rng.choice(NIVELES_EDU, n, p=[.25, .20, .40, .15]),
        "ocupacion": ocup,
        "ingreso_mensual": ingreso,
        "score_crediticio": score,
        "num_dependientes": np.where(ocup == "Estudiante", 0,
                                     np.clip(rng.poisson(1.2, n), 0, 5)),
        "fecha_registro": f_registro,
        "canal_adquisicion": rng.choice(CANALES, n, p=[.22, .18, .28, .14, .18]),
    })


def generar_polizas(cli):
    n = N_POLIZAS
    # Todo cliente tiene >=1 póliza; las extras favorecen a quienes ganan más
    peso = (cli["ingreso_mensual"] / cli["ingreso_mensual"].sum()).to_numpy()
    extra = rng.choice(len(cli), n - len(cli), p=peso)
    cidx = np.concatenate([np.arange(len(cli)), extra])
    rng.shuffle(cidx)
    c = cli.iloc[cidx].reset_index(drop=True)

    cob = rng.choice(COBERTURAS, n, p=P_COBERTURA)

    # La póliza inicia después del registro del cliente (y desde 2021)
    lo = np.maximum(c["fecha_registro"].to_numpy(dtype="datetime64[D]"),
                    np.datetime64("2021-01-01"))
    span = (np.datetime64("2025-06-30") - lo).astype(int)
    inicio = lo + (rng.random(n) * span).astype(int).astype("timedelta64[D]")
    inicio = pd.Series(pd.to_datetime(inicio))

    s_cob = pd.Series(cob)
    suma = np.round(s_cob.map(MED_SUMA).to_numpy()
                    * rng.lognormal(0, s_cob.map(SIG_SUMA).to_numpy()), -6)
    suma = np.maximum(suma, 10e6)

    edad = c["edad"].to_numpy()
    f_edad = np.where(np.isin(cob, ["Salud", "Vida"]), 1 + .012 * (edad - 40),
                      np.where(cob == "Auto", 1 + .01 * (35 - edad), 1.0))
    f_edad = np.clip(f_edad, .7, 1.6)
    prima = np.round(suma * s_cob.map(TASA_PRIMA).to_numpy() * f_edad
                     * rng.lognormal(0, .12, n), -3)

    # 18% de las pólizas sufre un incremento fuerte de prima (15%-40%)
    inc = np.where(rng.random(n) < .18, rng.uniform(15, 40, n), rng.normal(5, 3.5, n))
    inc = np.round(np.clip(inc, -4, 45), 1)
    prima_ant = np.round(prima / (1 + inc / 100), -3)

    return pd.DataFrame({
        "poliza_id": [f"POL-{i:06d}" for i in range(1, n + 1)],
        "cliente_id": c["cliente_id"].to_numpy(),
        "tipo_cobertura": cob,
        "fecha_inicio": inicio,
        "fecha_fin": inicio + pd.Timedelta(days=365),
        "suma_asegurada": suma,
        "prima_anual": prima,
        "prima_anterior": prima_ant,
        "incremento_prima_pct": inc,
        "forma_pago": rng.choice(FORMAS_PAGO, n, p=[.45, .15, .10, .30]),
        "canal_venta": rng.choice(CANALES, n, p=[.22, .18, .28, .14, .18]),
        "deducible_pct": rng.choice([0, 5, 10, 15], n, p=[.25, .35, .25, .15]),
        # columnas auxiliares (se eliminan antes de guardar)
        "_score": c["score_crediticio"].to_numpy(),
        "_ciudad": c["ciudad"].to_numpy(),
        "_registro": c["fecha_registro"].to_numpy(),
    })


def generar_reclamos(pol):
    n = N_RECLAMOS
    fin_eff = pol["fecha_fin"].clip(upper=FECHA_CORTE)
    exp_dias = (fin_eff - pol["fecha_inicio"]).dt.days.to_numpy()

    # Más reclamos en pólizas con más exposición, tipo Salud/Auto y score bajo
    w = (pol["tipo_cobertura"].map(FREQ_SINIESTRO).to_numpy() * exp_dias / 365
         * (1 + .5 * (pol["_score"].to_numpy() < 550)))
    pidx = rng.choice(len(pol), n, p=w / w.sum())
    p = pol.iloc[pidx].reset_index(drop=True)
    exp = exp_dias[pidx]
    cob = p["tipo_cobertura"].to_numpy()

    dias_ini = (rng.random(n) * exp).astype(int)       # días desde inicio de póliza
    f_sin = p["fecha_inicio"] + pd.to_timedelta(dias_ini, unit="D")

    tipo = np.empty(n, dtype=object)
    for c_, (lista, probs) in TIPOS_SINIESTRO.items():
        m = cob == c_
        tipo[m] = rng.choice(lista, m.sum(), p=probs)

    # --- FRAUDE: modelo logístico oculto ---------------------------------
    sc_z = (p["_score"].to_numpy() - 650) / 120
    logit = (1.3 * (dias_ini < 60)                       # siniestro muy temprano
             + .5 * np.isin(cob, ["Auto", "Hogar"])
             + .6 * (tipo == "Robo")
             - .45 * sc_z                                # score bajo => más riesgo
             + .35 * (p["canal_venta"].to_numpy() == "Web"))
    b0 = calibrar_intercepto(logit, TASA_FRAUDE)
    fraude = rng.random(n) < sigmoide(logit + b0)

    # --- Variables condicionadas al fraude (señales detectables) ---------
    med = pd.Series(cob).map(MED_RECLAMO).to_numpy()
    mult = np.array([MULT_TIPO.get((c_, t_), 1.0) for c_, t_ in zip(cob, tipo)])
    monto = med * mult * rng.lognormal(0, .7, n)
    monto = np.where(fraude, monto * rng.uniform(1.6, 3.2, n), monto)   # montos inflados
    monto = np.minimum(monto, .95 * p["suma_asegurada"].to_numpy())
    monto = np.maximum(np.round(monto, -3), 50_000)

    lag = np.where(fraude, rng.exponential(18, n), rng.exponential(4, n))
    lag = np.minimum(lag, np.where(fraude, 90, 30)).astype(int)         # reporte tardío
    f_rec = (f_sin + pd.to_timedelta(lag, unit="D")).clip(upper=fin_eff.to_numpy()[pidx])

    hora = np.clip(rng.normal(14, 4.5, n).round(), 0, 23).astype(int)
    noche = fraude & (rng.random(n) < .55)
    hora = np.where(noche, rng.choice([22, 23, 0, 1, 2, 3, 4, 5], n), hora)   # horario nocturno

    docs = np.where(fraude, np.clip(rng.poisson(2, n), 0, 10),
                    np.clip(rng.poisson(5.5, n), 1, 10))                  # pocos soportes

    canal = np.where(
        fraude,
        rng.choice(["App", "Call center", "Oficina", "Corredor"], n, p=[.20, .45, .15, .20]),
        rng.choice(["App", "Call center", "Oficina", "Corredor"], n, p=[.35, .30, .20, .15]))

    # --- Estado del reclamo ------------------------------------------------
    reciente = (f_rec >= FECHA_CORTE - pd.Timedelta(days=120)).to_numpy()
    p_rev = np.where(reciente, .5, 0.0)
    resto = 1 - p_rev
    p_rej = resto * np.where(fraude, .50, .15)
    p_par = resto * np.where(fraude, .10, .13)
    p_apr = resto - p_rej - p_par
    u = rng.random(n)
    estado = np.where(u < p_apr, "Aprobado",
             np.where(u < p_apr + p_par, "Pagado parcial",
             np.where(u < p_apr + p_par + p_rej, "Rechazado", "En revisión")))

    factor = np.where(estado == "Aprobado", rng.uniform(.85, 1.0, n),
             np.where(estado == "Pagado parcial", rng.uniform(.3, .7, n), 0.0))
    pagado = np.round(monto * factor, -3)
    pagado = np.where(estado == "En revisión", np.nan, pagado)   # aún sin pago

    dias_proc = np.where(fraude, rng.exponential(30, n) + 10,
                         rng.exponential(12, n) + 2).astype(int)
    f_res = (f_rec + pd.to_timedelta(dias_proc, unit="D")).clip(upper=FECHA_CORTE)
    f_res = f_res.where(estado != "En revisión")

    ciudad = np.where(rng.random(n) < .08, rng.choice(list(CIUDADES), n), p["_ciudad"].to_numpy())

    return pd.DataFrame({
        "reclamo_id": [f"REC-{i:06d}" for i in range(1, n + 1)],
        "poliza_id": p["poliza_id"].to_numpy(),
        "fecha_siniestro": f_sin,
        "fecha_reclamo": f_rec,
        "fecha_resolucion": f_res,
        "tipo_siniestro": tipo,
        "ciudad_siniestro": ciudad,
        "canal_reporte": canal,
        "hora_siniestro": hora,
        "num_documentos": docs,
        "monto_solicitado": monto,
        "monto_pagado": pagado,
        "estado_reclamo": estado,
        "sospecha_fraude": fraude.astype(int),
    })


def asignar_estado_polizas(pol, rec):
    """Calcula el churn (estado 'Cancelada') a partir de prima, reclamos y perfil."""
    n = len(pol)
    g = rec.groupby("poliza_id")
    n_recl = pol["poliza_id"].map(g.size()).fillna(0).to_numpy()
    n_rech = pol["poliza_id"].map(
        rec[rec["estado_reclamo"] == "Rechazado"].groupby("poliza_id").size()).fillna(0).to_numpy()
    ultimo = pol["poliza_id"].map(g["fecha_reclamo"].max())

    antig = (pol["fecha_inicio"] - pol["_registro"]).dt.days.to_numpy() / 365
    sc_z = (pol["_score"].to_numpy() - 650) / 120
    logit = (.06 * pol["incremento_prima_pct"].to_numpy()   # alza de prima
             + .55 * n_rech                                 # reclamos rechazados
             + .12 * n_recl
             - .30 * sc_z
             - .10 * antig
             + .30 * (pol["forma_pago"].to_numpy() == "Mensual")
             + .25 * (pol["canal_venta"].to_numpy() == "Web")
             - .25 * (pol["tipo_cobertura"].to_numpy() == "Vida")
             + rng.normal(0, .5, n))
    b0 = calibrar_intercepto(logit, TASA_CHURN)
    cancela = rng.random(n) < sigmoide(logit + b0)

    # La cancelación ocurre después del último reclamo y antes del fin/corte
    fin_eff = pol["fecha_fin"].clip(upper=FECHA_CORTE)
    dias_vig = (fin_eff - pol["fecha_inicio"]).dt.days.to_numpy()
    base_c = pol["fecha_inicio"] + pd.to_timedelta(30 + rng.random(n) * (dias_vig - 30), unit="D")
    tras = ultimo.fillna(pol["fecha_inicio"]) + pd.to_timedelta(rng.integers(5, 46, n), unit="D")
    f_cancel = base_c.where(base_c >= tras, tras).clip(upper=fin_eff)

    estado = np.where(
        cancela, "Cancelada",
        np.where(pol["fecha_fin"] <= FECHA_CORTE,
                 np.where(rng.random(n) < .65, "Renovada", "Vencida"),
                 np.where(rng.random(n) < .02, "Suspendida", "Activa")))

    motivo = np.full(n, None, dtype=object)
    inc = pol["incremento_prima_pct"].to_numpy()
    for i in np.flatnonzero(cancela):
        if inc[i] > 15 and rng.random() < .55:
            motivo[i] = "Precio"
        elif n_rech[i] > 0 and rng.random() < .50:
            motivo[i] = "Reclamo rechazado / insatisfacción"
        else:
            motivo[i] = rng.choice(
                ["Cambio de aseguradora", "Precio", "Falta de pago",
                 "Venta/pérdida del bien", "Otro"], p=[.30, .20, .25, .10, .15])

    out = pol.copy()
    out["estado"] = estado
    out["fecha_cancelacion"] = f_cancel.where(cancela)
    out["motivo_cancelacion"] = motivo
    out["_n_recl"], out["_n_rech"] = n_recl, n_rech
    return out


# ---------------------------------------------------------------------
# 4. INYECCIÓN DE SUCIEDAD
# ---------------------------------------------------------------------
def ensuciar_clientes(df):
    d, n = df.copy(), len(df)
    d["nombre"] = ensuciar_texto(d["nombre"], p=.20)
    d["genero"] = ensuciar_texto(d["genero"], p=.10, alias=ALIAS_GENERO, p_alias=.06)
    d["ciudad"] = poner_nulos(
        ensuciar_texto(d["ciudad"], p=.35, alias=ALIAS_CIUDAD, p_alias=.08), .03)
    for col in ["ocupacion", "estado_civil", "nivel_educativo", "canal_adquisicion"]:
        d[col] = ensuciar_texto(d[col], p=.20)
    d["ocupacion"] = poner_nulos(d["ocupacion"], .02)

    edad = poner_nulos(d["edad"], .015)
    i = idx_aleatorios(n, .004)
    edad.iloc[i] = rng.choice([-5, -1, -3, 0], len(i))          # edades imposibles
    i = idx_aleatorios(n, .004)
    edad.iloc[i] = rng.choice([150, 120, 200, 999], len(i))     # edades irreales
    d["edad"] = a_entero_nulo(edad)

    ing = poner_nulos(d["ingreso_mensual"], .05)
    i = idx_aleatorios(n, .008)
    ing.iloc[i] = 0                                             # ingreso cero
    i = idx_aleatorios(n, .005)
    ing.iloc[i] = ing.iloc[i] * rng.choice([50, 100, 1000], len(i))   # desproporcionados
    ing = ing.astype(object)
    for k in np.flatnonzero(ing.notna().to_numpy() & (rng.random(n) < .03)):
        ing.iloc[k] = "$" + f"{ing.iloc[k]:,.0f}".replace(",", ".")   # texto "$3.450.000"
    d["ingreso_mensual"] = ing

    sc = poner_nulos(d["score_crediticio"], .02)
    i = idx_aleatorios(n, .003)
    sc.iloc[i] = 9999                                           # código de error
    d["score_crediticio"] = a_entero_nulo(sc)

    d["fecha_registro"] = mezclar_fechas(d["fecha_registro"], .02, .015)
    return d


def ensuciar_polizas(df):
    d, n = df.copy(), len(df)

    # Llaves foráneas: huérfanas (cliente inexistente) y con formato roto
    cid = d["cliente_id"].astype(object).copy()
    i = idx_aleatorios(n, .004)
    cid.iloc[i] = [f"CLI-{x}" for x in rng.integers(90000, 99999, len(i))]
    i = idx_aleatorios(n, .01)
    cid.iloc[i] = [v.lower() if rng.random() < .5 else f" {v} " for v in cid.iloc[i]]
    d["cliente_id"] = cid

    d["tipo_cobertura"] = ensuciar_texto(d["tipo_cobertura"], .30, ALIAS_COBERTURA, .05)
    d["estado"] = ensuciar_texto(d["estado"], .25, ALIAS_ESTADO_POL, .05)
    d["forma_pago"] = ensuciar_texto(d["forma_pago"], .20)
    d["canal_venta"] = ensuciar_texto(d["canal_venta"], .20)
    d["motivo_cancelacion"] = ensuciar_texto(d["motivo_cancelacion"], .15)

    # Error lógico: fecha_fin anterior a fecha_inicio
    i = idx_aleatorios(n, .004)
    d.iloc[i, d.columns.get_loc("fecha_fin")] = (
        d["fecha_inicio"].iloc[i] - rng.integers(1, 365, len(i)).astype("timedelta64[D]")).to_numpy()

    pr = poner_nulos(d["prima_anual"], .015)
    i = idx_aleatorios(n, .004)
    pr.iloc[i] = 0
    i = idx_aleatorios(n, .003)
    pr.iloc[i] = pr.iloc[i] * 100
    i = idx_aleatorios(n, .002)
    pr.iloc[i] = -pr.iloc[i]
    d["prima_anual"] = a_entero_nulo(pr)

    d["prima_anterior"] = a_entero_nulo(poner_nulos(d["prima_anterior"], .02))
    d["suma_asegurada"] = a_entero_nulo(poner_nulos(d["suma_asegurada"], .01))
    d["incremento_prima_pct"] = poner_nulos(d["incremento_prima_pct"], .02)

    for col in ["fecha_inicio", "fecha_fin", "fecha_cancelacion"]:
        d[col] = mezclar_fechas(d[col], .015, .010)
    return d


def ensuciar_reclamos(df):
    d, n = df.copy(), len(df)

    pid = d["poliza_id"].astype(object).copy()
    i = idx_aleatorios(n, .003)
    pid.iloc[i] = [f"POL-{x}" for x in rng.integers(900000, 999999, len(i))]
    i = idx_aleatorios(n, .01)
    pid.iloc[i] = [v.lower() if rng.random() < .5 else f" {v} " for v in pid.iloc[i]]
    d["poliza_id"] = pid

    d["tipo_siniestro"] = ensuciar_texto(d["tipo_siniestro"], .25)
    d["canal_reporte"] = ensuciar_texto(d["canal_reporte"], .20)
    d["ciudad_siniestro"] = poner_nulos(
        ensuciar_texto(d["ciudad_siniestro"], .35, ALIAS_CIUDAD, .08), .03)
    d["estado_reclamo"] = ensuciar_texto(d["estado_reclamo"], .25, ALIAS_ESTADO_REC, .05)

    # Error lógico: siniestro anterior al inicio de la póliza
    i = idx_aleatorios(n, .005)
    d.iloc[i, d.columns.get_loc("fecha_siniestro")] = (
        d["fecha_siniestro"].iloc[i] - np.timedelta64(400, "D")).to_numpy()

    sol0 = d["monto_solicitado"].astype(float)
    sol = poner_nulos(sol0, .025)
    i = idx_aleatorios(n, .002)
    sol.iloc[i] = -sol.iloc[i]
    i = idx_aleatorios(n, .003)
    sol.iloc[i] = 0
    i = idx_aleatorios(n, .003)
    sol.iloc[i] = sol.iloc[i] * 100
    d["monto_solicitado"] = a_entero_nulo(sol)

    pag = poner_nulos(d["monto_pagado"], .02)       # (los 'En revisión' ya traen nulo legítimo)
    i = idx_aleatorios(n, .005)
    pag.iloc[i] = sol0.iloc[i] * rng.uniform(1.3, 3.0, len(i))   # pagado > solicitado
    d["monto_pagado"] = a_entero_nulo(pag)

    hora = d["hora_siniestro"].astype(float)
    i = idx_aleatorios(n, .003)
    hora.iloc[i] = rng.choice([24, 25, 99, -1], len(i))          # horas imposibles
    d["hora_siniestro"] = a_entero_nulo(hora)
    d["num_documentos"] = a_entero_nulo(poner_nulos(d["num_documentos"], .03))

    # Etiqueta de fraude con codificaciones mixtas y algunos desconocidos
    f = d["sospecha_fraude"].astype(object)
    i = idx_aleatorios(n, .10)
    es_si = (f.iloc[i] == 1).to_numpy()
    f.iloc[i] = np.where(es_si, rng.choice(["Sí", "SI", "S", "true"], len(i)),
                         rng.choice(["No", "NO", "N", "false"], len(i)))
    f.iloc[idx_aleatorios(n, .005)] = np.nan
    d["sospecha_fraude"] = f

    for col in ["fecha_siniestro", "fecha_reclamo", "fecha_resolucion"]:
        p_nulo = .04 if col == "fecha_reclamo" else .015
        d[col] = mezclar_fechas(d[col], p_nulo, .015)
    return d


# ---------------------------------------------------------------------
# 5. VALIDACIÓN DE PATRONES (sobre datos limpios, antes de ensuciar)
# ---------------------------------------------------------------------
def validar_patrones(pol, rec):
    churn = (pol["estado"] == "Cancelada")
    print(f"\n[Patrón churn] Tasa de cancelación: {churn.mean():.1%}")
    b = pd.cut(pol["incremento_prima_pct"], [-10, 10, 20, 50], labels=["<10%", "10-20%", ">20%"])
    print("  Churn por incremento de prima:\n  ",
          churn.groupby(b, observed=True).mean().round(3).to_dict())
    print("  Churn por # reclamos rechazados (0/1/2+):\n  ",
          churn.groupby(pol["_n_rech"].clip(upper=2)).mean().round(3).to_dict())

    fr = rec["sospecha_fraude"].astype(bool)
    print(f"\n[Patrón fraude] Tasa de sospecha: {fr.mean():.1%}")
    print("  Monto solicitado medio (fraude vs normal): "
          f"{rec.loc[fr, 'monto_solicitado'].mean():,.0f} vs {rec.loc[~fr, 'monto_solicitado'].mean():,.0f}")
    print("  Documentos medios (fraude vs normal): "
          f"{rec.loc[fr, 'num_documentos'].mean():.1f} vs {rec.loc[~fr, 'num_documentos'].mean():.1f}")


def resumen(nombre, df):
    print(f"\n{nombre}: {len(df):,} filas | duplicados exactos: {df.duplicated().sum():,} "
          f"({df.duplicated().mean():.1%}) | celdas nulas: {int(df.isna().sum().sum()):,}")


# ---------------------------------------------------------------------
# 6. EJECUCIÓN
# ---------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cli = generar_clientes()
    pol = generar_polizas(cli)
    rec = generar_reclamos(pol)
    pol = asignar_estado_polizas(pol, rec)

    validar_patrones(pol, rec)

    aux = [c for c in pol.columns if c.startswith("_")]
    cols_cli = list(cli.columns)
    cols_pol = [c for c in pol.columns if c not in aux and c != "estado"
                and c not in ("fecha_cancelacion", "motivo_cancelacion")]
    cols_pol += ["estado", "fecha_cancelacion", "motivo_cancelacion"]
    cols_rec = list(rec.columns)

    pol_limpia, rec_limpia = pol[cols_pol].copy(), rec[cols_rec].copy()

    if GUARDAR_REFERENCIA_LIMPIA:
        cli.to_csv(OUTPUT_DIR / "clientes_limpio_referencia.csv", index=False)
        pol_limpia.to_csv(OUTPUT_DIR / "polizas_limpio_referencia.csv", index=False)
        rec_limpia.to_csv(OUTPUT_DIR / "reclamos_limpio_referencia.csv", index=False)

    cli_raw = agregar_duplicados(ensuciar_clientes(cli))
    pol_raw = agregar_duplicados(ensuciar_polizas(pol_limpia))
    rec_raw = agregar_duplicados(ensuciar_reclamos(rec_limpia))

    archivos = {"clientes_raw.csv": cli_raw, "polizas_raw.csv": pol_raw, "reclamos_raw.csv": rec_raw}
    for nombre, df in archivos.items():
        df.to_csv(OUTPUT_DIR / nombre, index=False, encoding="utf-8")
        resumen(nombre, df)

    print(f"\nListo. Archivos guardados en: {OUTPUT_DIR.resolve()}")
    # En Google Colab, para descargarlos:
    #   from google.colab import files
    #   for f in archivos: files.download(str(OUTPUT_DIR / f))
    return archivos


if __name__ == "__main__":
    main()
    