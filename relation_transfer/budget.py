"""Durable per-call charging shared by spawn workers, independent of checkpoints."""
import os
import sqlite3
import time
from contextlib import contextmanager,closing
from pathlib import Path
from .contracts import require


class BudgetExhausted(RuntimeError): pass


class Budget:
    def __init__(self,path): self.path=str(path)

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=120,isolation_level=None)
        try:
            db.execute('PRAGMA synchronous=FULL')
            yield db
        finally:
            db.close()

    def create(self,c):
        require(not Path(self.path).exists(),'budget already exists')
        with self.connect() as db:
            db.executescript('CREATE TABLE budget (used INTEGER, cap INTEGER, wall REAL, wall_cap REAL, started REAL, owner INTEGER); CREATE TABLE charges (category TEXT PRIMARY KEY, count INTEGER); CREATE TABLE progress (task TEXT PRIMARY KEY, step INTEGER); CREATE TABLE control (aborted INTEGER); INSERT INTO control VALUES (0);')
            db.execute('INSERT INTO budget VALUES (0,?,0,?,NULL,NULL)',(c['global_max_repair_calls'],c['global_max_wall_seconds']))

    def start(self):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT started FROM budget').fetchone()
            require(row[0] is None,'unclean session: active wall duration unknown; fail closed, do not remove locks/ledger')
            db.execute('UPDATE budget SET started=?,owner=?',(time.time(),os.getpid())); db.execute('UPDATE control SET aborted=0'); db.commit()

    def check(self):
        with self.connect() as db:
            wall,cap,started=db.execute('SELECT wall,wall_cap,started FROM budget').fetchone()
            if db.execute('SELECT aborted FROM control').fetchone()[0]: raise RuntimeError('coordinator cancelled remaining work')
        if started is None: raise RuntimeError('no budget session')
        if wall+max(0,time.time()-started)>=cap: raise BudgetExhausted('active wall cap')

    def charge(self,category):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            used,cap,wall,wcap,started,_=db.execute('SELECT * FROM budget').fetchone()
            if db.execute('SELECT aborted FROM control').fetchone()[0]:
                db.rollback(); raise RuntimeError('coordinator cancelled remaining work')
            if started is None or used>=cap or wall+max(0,time.time()-started)>=wcap:
                db.rollback(); raise BudgetExhausted('Repair or active wall cap')
            db.execute('UPDATE budget SET used=used+1')
            db.execute('INSERT INTO charges VALUES (?,1) ON CONFLICT(category) DO UPDATE SET count=count+1',(category,)); db.commit()

    def abort(self):
        with self.connect() as db: db.execute('UPDATE control SET aborted=1')

    def progress(self,task,step):
        with self.connect() as db:
            db.execute('INSERT INTO progress VALUES (?,?) ON CONFLICT(task) DO UPDATE SET step=excluded.step',(task,step))

    def finish(self):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            started,owner=db.execute('SELECT started,owner FROM budget').fetchone()
            require(owner==os.getpid(),'only coordinator closes session')
            db.execute('UPDATE budget SET wall=wall+?,started=NULL,owner=NULL',(max(0,time.time()-started),)); db.commit()


def snapshot(path):
    uri=Path(path).resolve().as_uri()+'?mode=ro'
    with closing(sqlite3.connect(uri,uri=True)) as db:
        used,cap,wall,wcap,started,owner=db.execute('SELECT * FROM budget').fetchone()
        return dict(repair_calls=used,repair_cap=cap,active_wall_seconds=wall+(max(0,time.time()-started) if started else 0),wall_cap=wcap,session_owner=owner,
            charges=dict(db.execute('SELECT * FROM charges')),progress=dict(db.execute('SELECT * FROM progress')))
