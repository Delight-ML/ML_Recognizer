last_idx = df_proc.index[-1]
            # build X consistent with training: model trained on ['Open','High','Low','Close']
            X = df_proc[['Open','High','Low','Close']].iloc[-1:].values
            # predict prob and class
            # if model supports predict_proba:
            prob = None
            try:
                prob = float(model.predict_proba(X)[0][1])
            except Exception:
                # if model gives only class, assume 1.0 confidence for that class (not ideal, but fallback)
                pred = int(model.predict(X)[0])
                prob = 1.0

            pred_class = int(model.predict(X)[0])  # 1 = buy, -1 = sell in our training script
            # compute HTF trend alignment (H1)
            htf_trend = compute_htf_trend(df)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            price = float(df_proc['Close'].iloc[-1])

            # apply filters
            bar_index = len(df_proc) - 1
            if bar_index - last_signal_bar_index < COOLDOWN_BARS:
                # cooldown active
                last_len = len(df_proc)
                time.sleep(POLL_INTERVAL)
                continue

            # only accept BUY if HTF trend is UP (1), or if neutral allow
            if pred_class == 1 and prob >= PROB_THRESH and (htf_trend >= 0):
                log_signal(ts, "BUY", prob, price, reason=f"prob>{PROB_THRESH} & HTF={htf_trend}")
                last_signal_bar_index = bar_index

            # only accept SELL if HTF trend is DOWN (-1), or if neutral allow
            elif pred_class == -1 and prob >= PROB_THRESH and (htf_trend <= 0):
                log_signal(ts, "SELL", prob, price, reason=f"prob>{PROB_THRESH} & HTF={htf_trend}")
                last_signal_bar_index = bar_index

            last_len = len(df_proc)
            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as e:
            print("Predict loop error:", e)
            time.sleep(POLL_INTERVAL)

if name == "main":
    main_loop()