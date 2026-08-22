## On-chain artefacts (live, public)

> *Every claim about gas cost or on-chain verifiability in the paper can be
> checked at these URLs by anyone with a browser. No login or wallet required
> to read.*

| Artefact | Value | Sepolia Etherscan |
|---|---|---|
| Contract | `ZKKGVerifyV2` | [Contract page](https://sepolia.etherscan.io/address/0x22199e2BFCc9f47278892eEE3386f4Bd78F44bd3) |
| Address | `0x22199e2BFCc9f47278892eEE3386f4Bd78F44bd3` | [↗](https://sepolia.etherscan.io/address/0x22199e2BFCc9f47278892eEE3386f4Bd78F44bd3) |
| Deployment gas | 1,325,277 | — |
| Deployment cost | 0.00335966 ETH (testnet) | [↗](https://sepolia.etherscan.io/tx/0x5ab35aab0205073cb87af8aef07a1ec0c8e1904c03eae1463091abe035e6a51d) |
| Model registration | 138,684 gas, single-shot | — |
| Records | 30 `verifyAndLog`, all chain-verified | [Events tab](https://sepolia.etherscan.io/address/0x22199e2BFCc9f47278892eEE3386f4Bd78F44bd3#events) |
| Gas, first record | 281,162 (cold storage slots) | — |
| Gas, steady state | 246,952 | — |
| On-chain verification | 56,568 gas of that | — |
| Total cost | 0.01927360 ETH (testnet) | — |
| Solidity | `v0.8.20`, optimizer 200 runs | [Code tab](https://sepolia.etherscan.io/address/0x22199e2BFCc9f47278892eEE3386f4Bd78F44bd3#code) |

The deployment gas is the quickest way to tell the two contracts apart:
V1, which does *not* verify, costs 545,765. V2 costs 1,325,277.

### Verify the on-chain claims yourself (no setup, ~60 s)

1. Open the [contract page](https://sepolia.etherscan.io/address/0x22199e2BFCc9f47278892eEE3386f4Bd78F44bd3) and confirm **32 transactions** — one creation, one `registerModel`, and 30 `verifyAndLog`.
2. Open the [Read Contract tab](https://sepolia.etherscan.io/address/0x22199e2BFCc9f47278892eEE3386f4Bd78F44bd3#readContract). `getRecordCount` returns `30`. `getRecord(0)` returns the first stored record, immutably.
3. Open any record transaction below and click *Decode Input Data* to see the parameters the contract was called with, and the emitted event under **Logs**.

The decisive point is what the contract does *before* it writes: it re-derives the Fiat-Shamir challenge and checks the group relation against its own registry. An invalid proof reverts, so a stored record is an attestation by consensus rather than a claim by whoever sent the transaction.

### Every record, individually

<details>
<summary><b>Click to expand — direct Etherscan link for each on-chain record</b></summary>

**Contract creation** — [0x5ab35aab...a51d](https://sepolia.etherscan.io/tx/0x5ab35aab0205073cb87af8aef07a1ec0c8e1904c03eae1463091abe035e6a51d), 1,325,277 gas, block 11,542,020

**30 chain-verified records**

1. [0xd75cedea...d295](https://sepolia.etherscan.io/tx/0xd75cedeae8b49da06583bbdc218fd80f2b76a383cc215489540185470203d295) — 281,162 gas, block 11,542,022, triple `[1934, 104, 8021]`
2. [0xbb2fc32a...3cf5](https://sepolia.etherscan.io/tx/0xbb2fc32ac3254f4cce8b1fbcd90ec375d4d18c52695bb7920b4fc63697183cf5) — 246,962 gas, block 11,542,023, triple `[477, 34, 9265]`
3. [0x8488bba1...463e](https://sepolia.etherscan.io/tx/0x8488bba183cdff2b148cee82e17200683c618341b610a4e4e3d113eaf333463e) — 246,950 gas, block 11,542,024, triple `[13034, 203, 1811]`
4. [0xfcb742d1...1bb5](https://sepolia.etherscan.io/tx/0xfcb742d19476284285a22a2d0a65dadfb0bdfd39d45265a711c8e9f8c2e21bb5) — 246,938 gas, block 11,542,025, triple `[12692, 149, 2820]`
5. [0x4f880c60...7092](https://sepolia.etherscan.io/tx/0x4f880c608e30cc10c3ba8b2f133cc97ac61a0f2275fe2c73fe415cd5edc57092) — 246,962 gas, block 11,542,026, triple `[9182, 11, 669]`
6. [0x618c3d54...f49b](https://sepolia.etherscan.io/tx/0x618c3d547c704dabba374e4ae11a8bbb70b0651502f450e9eacf6a3ea906f49b) — 246,950 gas, block 11,542,027, triple `[5258, 6, 725]`
7. [0xe0a214ce...8020](https://sepolia.etherscan.io/tx/0xe0a214ce27a27bec99545bda8990a72b0861592c9e55126c2d93da1eb11d8020) — 246,950 gas, block 11,542,028, triple `[7835, 68, 5697]`
8. [0x803f4e04...7131](https://sepolia.etherscan.io/tx/0x803f4e0490fbae42b962e40ec3f1a6d56f84d4477fd90071c443b67967927131) — 246,962 gas, block 11,542,029, triple `[13108, 138, 10559]`
9. [0xb0ea49f1...cac4](https://sepolia.etherscan.io/tx/0xb0ea49f1a183fa3beb5a21b271971ddca0df12603cd5006a61d0e37ab1a2cac4) — 246,950 gas, block 11,542,030, triple `[9571, 79, 12531]`
10. [0xee164631...e3f3](https://sepolia.etherscan.io/tx/0xee1646312069b449fdace5571942e1bb524cc74009c20a3f4d46e2ea7dbae3f3) — 246,962 gas, block 11,542,031, triple `[1192, 189, 14406]`
11. [0x62b60674...3125](https://sepolia.etherscan.io/tx/0x62b6067400391d03dda2d42ae945a988c7840b7fe8620244c37ef71a96023125) — 246,950 gas, block 11,542,032, triple `[4471, 129, 13829]`
12. [0x0c547d5f...aa9b](https://sepolia.etherscan.io/tx/0x0c547d5fa72f076d09b1901ae19fee51d3e44cd5c3a101e956f15f4198ccaa9b) — 246,962 gas, block 11,542,033, triple `[6172, 89, 10717]`
13. [0x4c00e624...15f9](https://sepolia.etherscan.io/tx/0x4c00e6248d93de366cc055ec80454e14f79b4514613efa812df00f6722b915f9) — 246,962 gas, block 11,542,034, triple `[3106, 140, 13143]`
14. [0xf6142755...7db5](https://sepolia.etherscan.io/tx/0xf6142755949e1fa5c5edcc76952d2d570ef3209c5a7430a2f3ee6ee6d12d7db5) — 246,950 gas, block 11,542,035, triple `[10626, 79, 5390]`
15. [0xe0484f63...b761](https://sepolia.etherscan.io/tx/0xe0484f6367b4db6cdfcf6939e96cd35d499d5bdcee692f6bb65a71417669b761) — 246,938 gas, block 11,542,036, triple `[8305, 227, 6275]`
16. [0x4ab7b5d6...71ca](https://sepolia.etherscan.io/tx/0x4ab7b5d69b202b1c0ce0605ba30375723f06e7fa3b983e32784ff780415571ca) — 246,950 gas, block 11,542,037, triple `[12597, 3, 13140]`
17. [0x18548f65...4da8](https://sepolia.etherscan.io/tx/0x18548f65e6009a5994e2f203eee61dc7c3de566ff25ca4026a650359e6334da8) — 246,950 gas, block 11,542,038, triple `[5211, 89, 3580]`
18. [0x86dc7b58...5dba](https://sepolia.etherscan.io/tx/0x86dc7b58572f5bfe439c6e4f4a4f4c8a7d817c6027093b71e4571c8a1e615dba) — 246,962 gas, block 11,542,039, triple `[9819, 59, 10008]`
19. [0x744f18ca...e42b](https://sepolia.etherscan.io/tx/0x744f18ca235bd0c40aa415a9cd49aa586acc3e272a4b20b9d96be550aee5e42b) — 246,962 gas, block 11,542,040, triple `[8254, 188, 10416]`
20. [0x174e6a93...eccb](https://sepolia.etherscan.io/tx/0x174e6a93f2dadafc449813980a9dc5cf7c0f2451a0b2701e1601ba4cabececcb) — 246,950 gas, block 11,542,041, triple `[926, 102, 4477]`
21. [0x4b412373...b9a4](https://sepolia.etherscan.io/tx/0x4b412373548b15c588d82c69c0f327d8236b89008a505a4759500b1123b8b9a4) — 246,938 gas, block 11,542,042, triple `[1082, 227, 10311]`
22. [0xe49e9217...0d65](https://sepolia.etherscan.io/tx/0xe49e9217806820d6879a23577a0f0a9e9de06b66faab349aa657f93c718d0d65) — 246,962 gas, block 11,542,043, triple `[13081, 114, 4904]`
23. [0xd33e1ffc...8ed5](https://sepolia.etherscan.io/tx/0xd33e1ffc9347c9feb26214f9743aa362ac9635350b80916d06ca369d54488ed5) — 246,950 gas, block 11,542,044, triple `[11997, 102, 13950]`
24. [0x11343328...740e](https://sepolia.etherscan.io/tx/0x11343328971b81960c5a7ee48c987d59954a5fa80cebaa80b5d5cc0eded3740e) — 246,950 gas, block 11,542,045, triple `[9470, 38, 4438]`
25. [0x52627922...bfe9](https://sepolia.etherscan.io/tx/0x526279222af3a37895615e732bba5114d6eb431a09a63b99e6071595c6d8bfe9) — 246,950 gas, block 11,542,046, triple `[13923, 127, 12970]`
26. [0x46b9933b...4abc](https://sepolia.etherscan.io/tx/0x46b9933bbcd36591ed3308dc02c3c141738a0586bd8af1b7f62d8f15cb8c4abc) — 246,938 gas, block 11,542,047, triple `[10416, 124, 13914]`
27. [0x7095bfe8...0d21](https://sepolia.etherscan.io/tx/0x7095bfe8f5eaa6b09118e17104ad1b4290978f49553c6149913fe0ba4c430d21) — 246,950 gas, block 11,542,048, triple `[4314, 12, 9086]`
28. [0x17aa1494...3a38](https://sepolia.etherscan.io/tx/0x17aa1494ed8e97add702d00d412505a04e648d4d4f35da4939832701a5733a38) — 246,950 gas, block 11,542,049, triple `[7417, 89, 11881]`
29. [0x2cc8204b...2f15](https://sepolia.etherscan.io/tx/0x2cc8204b6304c68c2058ca868c1a916b980cc12c448f46e0bd13d99d939e2f15) — 246,950 gas, block 11,542,050, triple `[13056, 215, 981]`
30. [0x93306ec6...d2aa](https://sepolia.etherscan.io/tx/0x93306ec689a9f921b0b0c8f9397765c916fe02657be8a537e8f0a9526025d2aa) — 246,950 gas, block 11,542,051, triple `[4611, 85, 14342]`

</details>

An independent export of these transactions, taken from Etherscan rather than produced by this code, is in [`docs/etherscan_export_sepolia.csv`](docs/etherscan_export_sepolia.csv): 32 transactions, all `Success`, in 32 consecutive blocks.
