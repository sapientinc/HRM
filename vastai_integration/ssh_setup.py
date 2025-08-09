from asyncio import IncompleteReadError
from typing import Optional
import asyncssh


class SSHSession:
    def __init__(self, host, port=22, username=None, client_keys=None):
        self.host = host
        self.port = port
        self.username = username
        self.client_keys = client_keys
        self._proc: asyncssh.process.SSHClientProcess
        self._conn: asyncssh.connection.SSHClientConnection
        self.marker = "__CMD_DONE__"

    async def __aenter__(self):
        self._conn = await asyncssh.connect(
            self.host,
            port=self.port,
            username=self.username,
            client_keys=self.client_keys,
            known_hosts=None
        )
        # Start a shell; term_type gives you prompt‐style behavior
        self._proc = await self._conn.create_process(
             '/bin/bash --noprofile --norc -i',
             term_type='dumb'    # <-- this requests a PTY
        )  # :contentReference[oaicite:0]{index=0}
        
        # Do not repeat commands in the shell
        self._proc.stdin.write('stty -echo\n')
        # Don't print the shell header (e.g. 'bash-5.2# ') before every command
        self._proc.stdin.write("export PS1=''\n")
        # Write __CMD_DONE__ when you are done
        self._proc.stdin.write(f"echo {self.marker}\n")
        await self._proc.stdin.drain()
        # Prime the pump: wait for the shell config to finish
        await self._proc.stdout.readuntil(self.marker)
        return self

    async def run(self, cmd: str) -> str:
        """Run one command, return its output (sans trailing prompt)."""
        self._proc: asyncssh.process.SSHClientProcess
        # write the command, then echo a unique marker
        self._proc.stdin.write(f"{cmd} && echo {self.marker}\n")
        # send
        await self._proc.stdin.drain()

        # read everything until that marker shows up
        data: str = await self._proc.stdout.readuntil(self.marker)
        # strip off the marker line and return the rest
        lines = data.strip().splitlines()
        return '\n'.join(lines[:-1])

    async def __aexit__(self, exc_type, exc, tb):
        self._proc.stdin.write_eof()
        self._conn.close()
